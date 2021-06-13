
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

struct LinearElasticState <: AbstractMaterialState end

# define which state belongs to the material
initial_material_state(::LinearElastic) = LinearElasticState()

# for restricted stress states to create buffer
get_stress_type(::LinearElasticState) = SymmetricTensor{2,3,Float64,6}

# constitutive drivers generally operate in 3D 
# (we could specialize for lower dimensions if needed for performance)
"""
    material_response(m::LinearElastic, Δε::SymmetricTensor{2,3})

Return the stress tensor and the stress tangent for the given strain ε such that

```math
\\boldsymbol{\\sigma} = \\mathbf{E}^\\text{e} : \\Delta \\boldsymbol{\\varepsilon} .
```
No `MaterialState` is needed for the stress computation, thus if a state is handed over to `material_response`, the same state is returned.
"""
function material_response(m::LinearElastic, ε::SymmetricTensor{2,3}, state::LinearElasticState=LinearElasticState(), Δt=nothing; cache=nothing, options=nothing)
    σ = m.Eᵉ ⊡ ε
    return σ, m.Eᵉ, state
end