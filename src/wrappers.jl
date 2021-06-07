abstract type AbstractDim{dim} end

struct Dim{dim} <: AbstractDim{dim} end # generic
struct UniaxialStrain{dim} <: AbstractDim{dim} end # 1D uniaxial strain state
struct UniaxialStress{dim} <: AbstractDim{dim} end # 1D uniaxial stress state
struct PlaneStrain{dim} <: AbstractDim{dim} end # 2D plane strain state
struct PlaneStress{dim} <: AbstractDim{dim} end # 2D plane stress state

reduce_dim(A::Tensor{1,3}, ::AbstractDim{dim}) where dim = Tensor{1,dim}(i->A[i])
reduce_dim(A::Tensor{2,3}, ::AbstractDim{dim}) where dim = Tensor{2,dim}((i,j)->A[i,j])
reduce_dim(A::Tensor{4,3}, ::AbstractDim{dim}) where dim = Tensor{4,dim}((i,j,k,l)->A[i,j,k,l])
reduce_dim(A::SymmetricTensor{2,3}, ::AbstractDim{dim}) where dim = SymmetricTensor{2,dim}((i,j)->A[i,j])
reduce_dim(A::SymmetricTensor{4,3}, ::AbstractDim{dim}) where dim = SymmetricTensor{4,dim}((i,j,k,l)->A[i,j,k,l])

increase_dim(A::Tensor{1,dim,T}) where {dim, T} = Tensor{1,3}(i->(i <= dim ? A[i] : zero(T)))
increase_dim(A::Tensor{2,dim,T}) where {dim, T} = Tensor{2,3}((i,j)->(i <= dim && j <= dim ? A[i,j] : zero(T)))
increase_dim(A::SymmetricTensor{2,dim,T}) where {dim, T} = SymmetricTensor{2,3}((i,j)->(i <= dim && j <= dim ? A[i,j] : zero(T)))

# restricted strain states
function material_response(
    dim::Union{Dim{d}, UniaxialStrain{d}, PlaneStrain{d}},
    m::AbstractMaterial,
    Δε::SymmetricTensor{2,d,T,3},
    state::AbstractMaterialState,
    Δt = nothing;
    cache = nothing,
    options = Dict{Symbol, Any}(),
    ) where {d,T}

    Δε_3D = increase_dim(Δε)
    σ, dσdε, material_state = material_response(m, Δε_3D, state, Δt; cache=cache, options=options)

    return reduce_dim(σ, dim), reduce_dim(dσdε, dim), material_state
end

# restricted stress states
function material_response(
    dim::Union{UniaxialStress{d}, PlaneStress{d}},
    m::AbstractMaterial,
    Δε::AbstractTensor{2,d,T},
    state::AbstractMaterialState,
    Δt = nothing;
    cache =  get_cache(dim, get_stress_type(state)),
    options = Dict{Symbol, Any}(),
    ) where {d, T}

    tol = get(options, :plane_stress_tol, 1e-8)
    max_iter = get(options, :plane_stress_max_iter, 10)
    converged = false

    Δε_3D = increase_dim(Δε)

    oop_idxs = get_zero_indices(dim, Δε_3D)
    ip_idxs = get_nonzero_indices(dim, Δε_3D)
    
    Δε_voigt = cache.x_f
    fill!(Δε_voigt, 0.0)
    σ_mandel = cache.F
    J = cache.DF

    for _ in 1:max_iter
        σ, ∂σ∂ε, temp_state = material_response(m, Δε_3D, state, Δt; cache=cache, options=options)
        tomandel!(σ_mandel, σ)
        if norm(view(σ_mandel, oop_idxs)) < tol
            converged = true
            ∂σ∂ε_2D = fromvoigt(SymmetricTensor{4,d}, inv(inv(tovoigt(∂σ∂ε))[ip_idxs, ip_idxs]))
            return reduce_dim(σ, dim), ∂σ∂ε_2D, temp_state
        end
        tomandel!(J, ∂σ∂ε)
        Δε_voigt[oop_idxs] = -view(J, oop_idxs, oop_idxs) \ view(σ_mandel, oop_idxs)
        Δε_correction = frommandel(SymmetricTensor{2,3}, Δε_voigt)
        Δε_3D = Δε_3D + Δε_correction
    end
    error("Not converged. Could not find plane stress state.")
end

# out of plane / axis components for restricted stress cases
get_zero_indices(::PlaneStress{2}, ::SymmetricTensor{2,3}) = [3, 4, 5] # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::PlaneStress{2}, ::SymmetricTensor{2,3}) = [1, 2, 6] # for voigt/mandel format, do not use on tensor data!
get_zero_indices(::PlaneStress{2}, ::Tensor{2,3}) = [3, 4, 5, 7, 8] # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::PlaneStress{2}, ::Tensor{2,3}) = [1, 2, 6, 9] # for voigt/mandel format, do not use on tensor data!

get_zero_indices(::UniaxialStress{1}, ::SymmetricTensor{2,3}) = collect(2:6) # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::UniaxialStress{1}, ::SymmetricTensor{2,3}) = [1] # for voigt/mandel format, do not use on tensor data!
get_zero_indices(::UniaxialStress{1}, ::Tensor{2,3}) = collect(2:9) # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::UniaxialStress{1}, ::Tensor{2,3}) = [1] # for voigt/mandel format, do not use on tensor data!

# fallback in case there is no cache defined
struct PlaneStressCache{TF, TDF, TX}
    F::TF
    DF::TDF
    x_f::TX
end

# generic fallback, Materials without field σ need to define it
get_stress_type(state::AbstractMaterialState) = typeof(state.σ)

# fallback for optional cache argument
function get_cache(::Union{UniaxialStress, PlaneStress}, ::Type{TT}) where {dim, T, TT<:AbstractTensor{2,dim,T}}
    M = Tensors.n_components(Tensors.get_base(TT))
    cache = PlaneStressCache(Vector{T}(undef,M), Matrix{T}(undef,M,M), Vector{T}(undef,M))
    return cache
end
