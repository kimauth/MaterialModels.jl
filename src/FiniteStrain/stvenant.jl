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

strainmeasure(::StVenant) = RightCauchyGreen()

struct StVenantState <: AbstractMaterialState
end

function initial_material_state(::StVenant)
    return StVenantState()
end

function StVenant(; λ::T, μ::T) where T
    return StVenant(λ, μ)
end

function _SPK(mp::StVenant, C::SymmetricTensor{2,3})
    I = one(C)
    return 0.5*mp.λ*(tr(C) - 3)*I + mp.μ*(C - I)
end

function material_response(mp::StVenant, C::SymmetricTensor{2,3}, state::StVenantState = StVenantState(), 
                           Δt=nothing; cache=nothing, options=nothing)
                           
    ∂S∂C, S =  gradient((C) -> _SPK(mp, C), C, :all)
    return S, ∂S∂C, StVenantState()
end


