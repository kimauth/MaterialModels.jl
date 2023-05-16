"""
    StVenant(; E, ν)

Hyperelastic material
#Arguments
- `λ::Float64`: Lamé parameter
- `μ::Float64`: Lamé parameter (shear modulus)
"""

struct StVenant <: AbstractMaterial
    λ::Float64
    μ::Float64
end

struct StVenantState <: AbstractMaterialState
end

function initial_material_state(::StVenant)
    return StVenantState()
end

function StVenant(; λ::T, μ::T) where T
    return StVenant(λ, μ)
end

strainmeasure(::StVenant) = RightCauchyGreen()

function ψ(mp::StVenant, C)
    (; μ, λ) = mp
    I = one(C)
    E = (C - I)/2
    psi = 1/2 * λ * tr(E)^2 + μ * E ⊡ E
    return psi
end

function material_response(mp::StVenant, C::SymmetricTensor{2,3}, state::StVenantState = StVenantState(), 
                           Δt=nothing; cache=nothing, options=nothing)
                           
    μ = mp.μ
    λ = mp.λ
    I = one(SymmetricTensor{2,3})

    S = λ/2 * (tr(C) - 3)*I + μ*(C-I)
    ∂S∂C = λ*(I⊗I) + μ*(otimesu(I,I) + otimesl(I,I))
    ∂S∂C = symmetric(∂S∂C)

    return S, ∂S∂C, StVenantState()
end


