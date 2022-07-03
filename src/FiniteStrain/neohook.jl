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

struct NeoHookState <: AbstractMaterialState
end

function initial_material_state(::NeoHook)
    return NeoHookState()
end

function NeoHook(; λ::T, μ::T) where T

    return NeoHook(λ, μ)
end

strainmeasure(::NeoHook) = RightCauchyGreen()

function ψ(mp::NeoHook, C::SymmetricTensor{2,3})
    J = sqrt(det(C))
    I = tr(C)
    return mp.μ/2 * (I-3) - mp.μ*log(J) + mp.λ/2 * log(J)^2
end

function material_response(mp::NeoHook, C::SymmetricTensor{2,3}, state::NeoHookState = NeoHookState(), 
                           Δt=nothing, cache=nothing, args...; options=nothing)
    invC = inv(C)
    J = sqrt(det(C))
    S = mp.μ*(one(SymmetricTensor{2,3}) - inv(C)) + mp.λ*log(J)*inv(C)
    ∂S∂C = mp.λ*(invC⊗invC) + 2*(mp.μ-mp.λ*log(J))*otimesu(invC,invC)
    ∂S∂C = symmetric(∂S∂C)
    return S, ∂S∂C, NeoHookState()
end
