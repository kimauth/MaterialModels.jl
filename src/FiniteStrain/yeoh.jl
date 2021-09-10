"""
    Yeoh(; λ, μ, c₂, c₃)

Hyper elastic material
#Arguments
- `λ::Float64`: Lamé parameter
- `μ::Float64`: Lamé parameter (shear modulus)
- `c₂::Float64`: Material constant
- `c₃::Float64`: Material constant
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
    return Yeoh(λ, μ, c₂, c₃)
end

strainmeasure(::Yeoh) = RightCauchyGreen()

function ψ(mp::Yeoh, C::SymmetricTensor{2,3})
    J = sqrt(det(C))
    I = tr(C)
    return mp.μ/2 * (I-3) + mp.c₂*(I-3)^2 + mp.c₃*(I-3)^3 - mp.μ*log(J) + mp.λ/2 * log(J)^2
end

function material_response(mp::Yeoh, C::SymmetricTensor{2,3}, state::YeohState = YeohState(), 
                           Δt=nothing; cache=nothing, options=nothing)
    invC=inv(C)
    detC=det(C); 
    J=sqrt(detC);

    dlnJdc = 1/2*invC
    d2lnJdcdc = 1/2*inv(otimesu(-C,C))

    𝐈 = one(C)
    Ic = tr(C)

    S = 2*(mp.μ/2*𝐈 + 2*mp.c₂*(Ic-3)*𝐈 + 3*mp.c₃*(Ic-3)^2*𝐈 - mp.μ*dlnJdc + mp.λ* log(J) * dlnJdc)
    ∂S∂E = 4 * (2*mp.c₂*(𝐈 ⊗ 𝐈) + 6*mp.c₃*(Ic-3)*(𝐈 ⊗ 𝐈) - mp.μ*d2lnJdcdc + mp.λ*(log(J)*d2lnJdcdc + (dlnJdc ⊗ dlnJdc)))
    ∂S∂E = symmetric(∂S∂E)
    # TODO use transform tangents from traits branch

    return S, ∂S∂E, YeohState()
end

