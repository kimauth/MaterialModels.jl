"""
    NeoHook(; E, ν)

Hyperelastic material
#Arguments
- `λ::Float64`: Lamé parameter
- `μ::Float64`: Lamé parameter (shear modulus)
"""

struct NeoHook <: AbstractMaterial
    λ::Float64
    μ::Float64
end

strainmeasure(::NeoHook) = RightCauchyGreen()

struct NeoHookState <: AbstractMaterialState
end

function initial_material_state(::NeoHook)
    return NeoHookState()
end

function NeoHook(; λ::T, μ::T) where T

    return NeoHook(λ, μ)
end

function ψ(mp::NeoHook, C::SymmetricTensor{2,3})
    J = sqrt(det(C))
    I = tr(C)
    return mp.μ/2 * (I-3) - mp.μ*log(J) + mp.λ/2 * log(J)^2
end

function material_response(mp::NeoHook, C::SymmetricTensor{2,3}, state::NeoHookState = NeoHookState(), 
                           Δt=nothing; cache=nothing, options=nothing)
    ∂²Ψ∂C², ∂Ψ∂C, _ =  hessian((C) -> ψ(mp, C), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C, NeoHookState()
end
