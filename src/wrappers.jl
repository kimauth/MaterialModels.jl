abstract type AbstractDim end

struct UniaxialStrain <: AbstractDim end # 1D uniaxial strain state
struct UniaxialStress <: AbstractDim end # 1D uniaxial stress state
struct PlaneStrain    <: AbstractDim end # 2D plane strain state
struct PlaneStress    <: AbstractDim end # 2D plane stress state
struct ThreeD         <: AbstractDim end # 3D

getdim(::UniaxialStrain) = 1
getdim(::UniaxialStress) = 1
getdim(::PlaneStrain) = 2
getdim(::PlaneStress) = 2
getdim(::ThreeD) = 3

reduce_dim(A::Tensor{1,3}, d::AbstractDim) = Tensor{1,getdim(d)}(i->A[i])
reduce_dim(A::Tensor{2,3}, d::AbstractDim) = Tensor{2,getdim(d)}((i,j)->A[i,j])
reduce_dim(A::Tensor{4,3}, d::AbstractDim) = Tensor{4,getdim(d)}((i,j,k,l)->A[i,j,k,l])
reduce_dim(A::SymmetricTensor{2,3}, d::AbstractDim) = SymmetricTensor{2,getdim(d)}((i,j)->A[i,j])
reduce_dim(A::SymmetricTensor{4,3}, d::AbstractDim) = SymmetricTensor{4,getdim(d)}((i,j,k,l)->A[i,j,k,l])

increase_dim(A::Tensor{1,dim,T}) where {dim,T} = Tensor{1,3}(i->(i <= dim ? A[i] : zero(T)))
increase_dim(A::Tensor{2,dim,T}) where {dim,T} = Tensor{2,3}((i,j)->(i <= dim && j <= dim ? A[i,j] : zero(T)))
increase_dim(A::SymmetricTensor{2,dim,T}) where {dim,T} = SymmetricTensor{2,3}((i,j)->(i <= dim && j <= dim ? A[i,j] : zero(T)))

# 3d materials pass through
function material_response(
    ::ThreeD,
    m::AbstractMaterial,
    Δε::AbstractTensor{2,3,T},
    state::AbstractMaterialState,
    Δt = nothing;
    cache = nothing,
    options = Dict{Symbol, Any}(),
    ) where {T}

    return material_response(m, Δε, state, Δt; cache=cache, options=options)

end

# restricted strain states
function material_response(
    dimstate::Union{UniaxialStrain, PlaneStrain},
    m::AbstractMaterial,
    Δε::AbstractTensor{2,d,T},
    state::AbstractMaterialState,
    Δt = nothing;
    cache = nothing,
    options = Dict{Symbol, Any}(),
    ) where {d,T}

    @assert(getdim(dimstate) == d)

    Δε_3D = increase_dim(Δε)
    σ, dσdε, material_state = material_response(m, Δε_3D, state, Δt; cache=cache, options=options)

    return reduce_dim(σ, dimstate), reduce_dim(dσdε, dimstate), material_state
end

# restricted stress states
function material_response(
    dimstate::Union{UniaxialStress, PlaneStress},
    m::AbstractMaterial,
    Δε::AbstractTensor{2,d,T},
    state::AbstractMaterialState,
    Δt = nothing;
    cache =  get_cache(m),
    options = Dict{Symbol, Any}(),
    ) where {d, T}
    
    @assert(getdim(dimstate) == d)

    tol = get(options, :plane_stress_tol, 1e-8)
    max_iter = get(options, :plane_stress_max_iter, 10)
    converged = false

    Δε_3D = increase_dim(Δε)
    
    zero_idxs = get_zero_indices(dimstate, Δε_3D)
    nonzero_idxs = get_nonzero_indices(dimstate, Δε_3D)
    
    Δε_voigt = cache.x_f
    fill!(Δε_voigt, 0.0)
    σ_mandel = cache.F
    J = cache.DF
    
    for _ in 1:max_iter
        σ, ∂σ∂ε, temp_state = material_response(m, Δε_3D, state, Δt; cache=cache, options=options)
        tomandel!(σ_mandel, σ)
        if norm(view(σ_mandel, zero_idxs)) < tol
            converged = true
            ∂σ∂ε_2D = fromvoigt(SymmetricTensor{4,d}, inv(inv(tovoigt(∂σ∂ε))[nonzero_idxs, nonzero_idxs]))
            return reduce_dim(σ, dimstate), ∂σ∂ε_2D, temp_state
        end
        tomandel!(J, ∂σ∂ε)
        fill!(Δε_voigt, 0.0)
        Δε_voigt[zero_idxs] = -view(J, zero_idxs, zero_idxs) \ view(σ_mandel, zero_idxs)
        Δε_correction = frommandel(SymmetricTensor{2,3}, Δε_voigt)
        Δε_3D = Δε_3D + Δε_correction
    end
    error("Not converged. Could not find plane stress state.")
end

# out of plane / axis components for restricted stress cases
get_zero_indices(::PlaneStress, ::SymmetricTensor{2,3}) = [3, 4, 5] # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::PlaneStress, ::SymmetricTensor{2,3}) = [1, 2, 6] # for voigt/mandel format, do not use on tensor data!
get_zero_indices(::PlaneStress, ::Tensor{2,3}) = [3, 4, 5, 7, 8] # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::PlaneStress, ::Tensor{2,3}) = [1, 2, 6, 9] # for voigt/mandel format, do not use on tensor data!

get_zero_indices(::UniaxialStress, ::SymmetricTensor{2,3}) = collect(2:6) # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::UniaxialStress, ::SymmetricTensor{2,3}) = [1] # for voigt/mandel format, do not use on tensor data!
get_zero_indices(::UniaxialStress, ::Tensor{2,3}) = collect(2:9) # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::UniaxialStress, ::Tensor{2,3}) = [1] # for voigt/mandel format, do not use on tensor data!

# fallback in case there is no cache defined
struct PlaneStressCache{TF, TDF, TX}
    F::TF
    DF::TDF
    x_f::TX
end

# generic fallback, Materials without field σ need to define it
get_stress_type(state::AbstractMaterialState) = typeof(state.σ)

# fallback for optional cache argument
function get_cache(m::AbstractMaterial)
    state = initial_material_state(m)
    stress_type = get_stress_type(state)
    T = eltype(stress_type)
    M = Tensors.n_components(Tensors.get_base(stress_type))
    cache = PlaneStressCache(Vector{T}(undef,M), Matrix{T}(undef,M,M), Vector{T}(undef,M))
    return cache
end
