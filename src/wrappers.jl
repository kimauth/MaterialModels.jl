abstract type AbstractDim{dim} end

struct Dim{dim} <: AbstractDim{dim} end # generic
struct UniaxialStrain{dim} <: AbstractDim{dim} end # 1D uniaxial strain state
struct UniaxialStress{dim} <: AbstractDim{dim} end # 1D uniaxial stress state
struct PlaneStrain{dim} <: AbstractDim{dim} end # 2D plane strain state
struct PlaneStress{dim} <: AbstractDim{dim} end # 2D plane stress state
# constructors without dim parameter
UniaxialStrain() = UniaxialStrain{1}()
UniaxialStress() = UniaxialStress{1}()
PlaneStrain() = PlaneStrain{2}()
PlaneStress() = PlaneStress{2}()

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
    Δε::AbstractTensor{2,d,T},
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
    cache =  get_cache(m),
    options = Dict{Symbol, Any}(),
    ) where {d, T}
    
    tol = get(options, :plane_stress_tol, 1e-8)
    max_iter = get(options, :plane_stress_max_iter, 10)
    converged = false

    Δε_3D = increase_dim(Δε)
    
    zero_idxs = get_zero_indices(dim, Δε_3D)
    nonzero_idxs = get_nonzero_indices(dim, Δε_3D)
    
    Δε_voigt = zeros(Float64, 6)
    fill!(Δε_voigt, 0.0)
    #σ_mandel = cache.F
    #J = cache.DF
    
    for _ in 1:max_iter
        σ, ∂σ∂ε, temp_state = material_response(m, Δε_3D, state, Δt; cache=cache, options=options)
        σ_mandel = tomandel2(dim, σ)
        if norm(view(σ_mandel, zero_idxs)) < tol
            converged = true
            ∂σ∂ε_2D = fromvoigt(SymmetricTensor{4,d}, inv(inv(tovoigt(∂σ∂ε))[nonzero_idxs, nonzero_idxs]))
            return reduce_dim(σ, dim), ∂σ∂ε_2D, temp_state
        end
        J = tomandel2(∂σ∂ε)
        fill!(Δε_voigt, 0.0)
        Δε_temp = -view(J, zero_idxs, zero_idxs) \ view(σ_mandel, zero_idxs)
        @show  typeof(Δε_temp)
        Δε_voigt[zero_idxs] = Δε_temp
        Δε_correction = frommandel(SymmetricTensor{2,3}, Δε_voigt)
        Δε_3D = Δε_3D + Δε_correction
    end
    error("Not converged. Could not find plane stress state.")
end

function testaa()
    m = LinearElastic(E=200e3, ν=0.3)

    state = initial_material_state(m)

    ε = rand(SymmetricTensor{2,2})

    a,b,c = material_response(PlaneStress(), m, ε, state)
end

function tomandel2(dim::PlaneStress, A::SymmetricTensor{2, 3}) 
    @SVector [A[1,1], A[2,2], A[3,3], √2A[2,3], √2A[1,3], √2A[1,2]]
end

function tomandel2(A::SymmetricTensor{4, 3}) 
    @SMatrix [A[1,1,1,1] A[1,1,2,2] A[1,1,3,3] √2*A[1,1,2,3] √2*A[1,1,1,3] √2*A[1,1,1,2]; 
              A[2,2,1,1] A[2,2,2,2] A[2,2,3,3] √2*A[2,2,2,3] √2*A[2,2,1,3] √2*A[2,2,1,2];
              A[3,3,1,1] A[3,3,2,2] A[3,3,3,3] √2*A[3,3,2,3] √2*A[3,3,1,3] √2*A[3,3,1,2];
              √2*A[2,3,1,1] √2*A[2,3,2,2] √2*A[2,3,3,3] 2*A[2,3,2,3] 2*A[2,3,1,3] 2*A[2,3,1,2];
              √2*A[1,3,1,1] √2*A[1,3,2,2] √2*A[1,3,3,3] 2*A[1,3,2,3] 2*A[1,3,1,3] 2*A[1,3,1,2];
              √2*A[1,2,1,1] √2*A[1,2,2,2] √2*A[1,2,3,3] 2*A[1,2,2,3] 2*A[1,2,1,3] 2*A[1,2,1,2]]
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
function get_cache(m::AbstractMaterial)
    state = initial_material_state(m)
    stress_type = get_stress_type(state)
    T = eltype(stress_type)
    M = Tensors.n_components(Tensors.get_base(stress_type))
    cache = PlaneStressCache(Vector{T}(undef,M), Matrix{T}(undef,M,M), Vector{T}(undef,M))
    return cache
end
