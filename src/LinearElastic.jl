
"""
    LinearElastic(E, ν)

Isotropic linear elasticity.
# Arguments
- `E::Float64`: Young's modulus
- `ν::Float64`: Poisson's ratio
"""
struct LinearElastic <: AbstractMaterial
    # parameters
    E::Float64 # Young's modulus
    ν::Float64 # Poisson's ratio

    # precomputed elastic stiffness tensor
    Eᵉ::SymmetricTensor{4,3,Float64,36}

    LinearElastic(E::Float64, ν::Float64) = new(E, ν, elastic_tangent_3D(E, ν))
end

# Let's always supply a constructor with keyword arguments
LinearElastic(;E::Float64, ν::Float64) = LinearElastic(E, ν)

function elastic_tangent_3D(E::T, ν::T) where T
    λ = E*ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    δ(i,j) = i == j ? 1.0 : 0.0
    f = (i,j,k,l) -> λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))

    C = SymmetricTensor{4, 3, T}(f)
    return C
end

# TODO not sure if we could hardcode Float64 here - do we want to differentiate states?
struct ElasticState{dim, T, M} <: AbstractMaterialState
    σ::SymmetricTensor{2,dim,T,M} # stress
end

# TODO could these be automatically generated?
Base.zero(::Type{ElasticState{dim,T,M}}) where {dim,T,M} = ElasticState(zero(SymmetricTensor{2,dim,T,M}))

# define which state belongs to the material
initial_material_state(::LinearElastic) = zero(ElasticState{3,Float64,6})

# constitutive drivers generally operate in 3D 
# (we could specialize for lower dimensions if needed for performance)
"""
    material_response(m::LinearElastic, Δε::SymmetricTensor{2,3}, state::ElasticState{3})

Return the stress tensor, stress tangent and the new `MaterialState` for the given strain step Δε such that

```math
\\boldsymbol{\\sigma} = \\mathbf{E}^\\text{e} : \\Delta \\boldsymbol{\\varepsilon} .
```
"""
function material_response(m::LinearElastic, Δε::SymmetricTensor{2,3}, state::ElasticState{3}, Δt=nothing; cache=nothing, options=nothing)
    Δσ = m.Eᵉ ⊡ Δε
    σ = state.σ + Δσ
    return σ, m.Eᵉ, ElasticState(σ)
end

"""
    TransversalLinearElastic(E, ν)

Transversal isotropic linear elasticity.
# Arguments
- `λ::Float64`: 1. Lame Parameters
- `μ::Float64`: 2. Lame Parameters
"""
struct TransversalLinearElastic  <: AbstractMaterial
    # parameters
    E::Float64 # Young's modulus
    ν::Float64 # Poisson's ratio
    α₁::Float64
    α₂::Float64
    α₃::Float64

    # precomputed elastic stiffness tensor
    Eᵉ::SymmetricTensor{4,3,Float64,36}

    TransversalLinearElastic(E::Float64, 
                             ν::Float64, 
                             α₁::Float64, 
                             α₂::Float64,
                             α₃::Float64, 
                             𝐀::Vec{3,Float64}) = new(E, ν, α₁, α₂, α₃, transversal_elastic_tangent_3D(E, ν, α₁, α₂, α₃, 𝐀⊗𝐀))
end

# Let's always supply a constructor with keyword arguments
TransversalLinearElastic(;E::Float64, ν::Float64, α₁::Float64, α₂::Float64, α₃::Float64, 𝐀::Vec{3,Float64}) = TransversalLinearElastic(E, ν, α₁, α₂, α₃, 𝐀)

function transversal_elastic_tangent_3D(E::T, ν::T, α₁::T, α₂::T, α₃::T, M::Tensor{2,3,T}) where T
    λ = E*ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    δ(i,j) = i == j ? 1.0 : 0.0
    f_iso  = (i,j,k,l) -> λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))
    f_tra = (i,j,k,l) -> α₁*(δ(i,j)*M[k,l] + M[i,j]*δ(k,l)) + 2α₂*(δ(i,j)*M[k,l] + M[i,j]*δ(k,l)) + α₃*M[i,j]*M[k,l]

    C = SymmetricTensor{4, 3, T}(f_iso + f_tra)
    return C
end

# TODO could these be automatically generated?
Base.zero(::Type{ElasticState{dim,T,M}}) where {dim,T,M} = ElasticState(zero(SymmetricTensor{2,dim,T,M}))

# define which state belongs to the material
initial_material_state(::LinearElastic) = zero(ElasticState{3,Float64,6})

# constitutive drivers generally operate in 3D 
# (we could specialize for lower dimensions if needed for performance)
"""
    material_response(m::LinearElastic, Δε::SymmetricTensor{2,3}, state::ElasticState{3})

Return the stress tensor, stress tangent and the new `MaterialState` for the given strain step Δε such that

```math
\\boldsymbol{\\sigma} = \\mathbf{E}^\\text{e} : \\Delta \\boldsymbol{\\varepsilon} .
```
"""
function material_response(m::LinearElastic, Δε::SymmetricTensor{2,3}, state::ElasticState{3}, Δt=nothing; cache=nothing, options=nothing)
    Δσ = m.Eᵉ ⊡ Δε
    σ = state.σ + Δσ
    return σ, m.Eᵉ, ElasticState(σ)
end
