"""
    Yeoh(; λ, μ, c_2, c_3)

Hyper elastic material, Yeoh
"""
struct Yeoh <: AbstractMaterial
    λ::Float64
    μ::Float64
    c₂::Float64
    c₃::Float64
end

struct YeohState <: AbstractMaterialState
end

function initial_material_state(::Yeoh)
    return YeohState()
end

function Yeoh(; λ::T, μ::T, c₂::T, c₃::T) where T <: AbstractFloat
    return MatYeoh(λ, μ, c₂, c₃)
end

function ψ(mp::MatYeoh, C::SymmetricTensor{2,3})
    J = sqrt(det(C))
    I = tr(C)
    return mp.μ/2 * (I-3) + mp.c₂*(I-3)^2 + mp.c₃*(I-3)^3 - mp.μ*log(J) + mp.λ/2 * log(J)^2
end

function material_response(mp::MatYeoh, C::SymmetricTensor{2,3})
    ∂²Ψ∂C², ∂Ψ∂C, _ =  hessian((C) -> ψ(mp, C), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C, YeohState()
end

