"""
    NeoHook(; λ, μ)

Hyper elastic material, Neo-Hook
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

function NeoHook(; E::T, ν::T) where T
    λ = (E*ν) / ((1+ν) * (1 - ν))
    μ = E / (2(1+ν))

    return NeoHook(λ, μ)
end

function ψ(mp::NeoHook, C::SymmetricTensor{2,3})
    J = sqrt(det(C))
    I = tr(C)
    return mp.μ/2 * (I-3) - mp.μ*log(J) + mp.λ/2 * log(J)^2
end

function material_response(mp::NeoHook, C::SymmetricTensor{2,3})
    ∂²Ψ∂C², ∂Ψ∂C, _ =  hessian((C) -> ψ(mp, C), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C, NeoHookState()
end
