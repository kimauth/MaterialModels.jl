"""
    StVenant(; E, ν)

Hyper elastic material, StVenant
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

function StVenant(; E::T, ν::T) where T
    λ = (E*ν) / ((1+ν) * (1 - ν))
    μ = E / (2(1+ν))

    return StVenant(λ, μ)
end

function ψ(mp::StVenant, C::SymmetricTensor{2,3})
    I = one(C)
    return 0.5*λ*(tr(C) - 3)*I + mp.μ*(C - I)
end

function material_response(mp::StVenant, C::SymmetricTensor{2,3})
    ∂²Ψ∂C², ∂Ψ∂C, _ =  hessian((C) -> ψ(mp, C), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C, StVenantState()
end
