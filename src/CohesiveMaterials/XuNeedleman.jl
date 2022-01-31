"""
    XuNeedleman(σₘₐₓ, τₘₐₓ, Φₙ, Φₜ, Δₙˢ)

Xu-Needleman traction-separation law.[^Xu1993] This is a commonly used cohesive law for brittle fracture, 
however it is revertible and thus not suited for unloading. In 3D, it is isotropic within the cohesive plane.

[^Xu1993]: Xu, X. P., & Needleman, A. (1993). Void nucleation by inclusion debonding in a crystal matrix. Modelling and Simulation in Materials Science and Engineering, 1(2), 111–132. [https://doi.org/10.1088/0965-0393/1/2/001](https://doi.org/10.1088/0965-0393/1/2/001)

# Arguments
- `σₘₐₓ::Float64`: cohesive normal strength
- `τₘₐₓ::Float64`: cohesive tangential strength (in-plane strength)
- `Φₙ::Float64`: normal work of separation
- `Φₜ::Float64`: tangential work of separation
- `Δₙˢ`: normal separation after complete shear separation under the condition of zero normal tension

For convenience the following functions can be used to convert parameters:
- [`xu_needleman_Φₙ`](@ref)
- [`xu_needleman_Φₜ`](@ref)
- [`xu_needleman_σₘₐₓ`](@ref)
- [`xu_needleman_τₘₐₓ`](@ref)
"""
struct XuNeedleman <: AbstractMaterial
    Φₙ::Float64
    Φₜ::Float64
    δₙ::Float64
    δₜ::Float64
    Δₙˢ::Float64
end

function XuNeedleman(; σₘₐₓ::Float64, τₘₐₓ::Float64, Φₙ::Float64, Φₜ::Float64, Δₙˢ::Float64)
    δₙ = Φₙ / (exp(1.0) * σₘₐₓ)
    δₜ = Φₜ / (sqrt(0.5exp(1.0)) * τₘₐₓ)
    return XuNeedleman(Φₙ, Φₜ, δₙ, δₜ, Δₙˢ)
end

# help converting between different possible material parameters
"""
    xu_needleman_Φₙ(σₘₐₓ, δₙ)

Compute the normal work of separation `Φₙ` for the Xu-Needleman cohesive law based on the cohesive normal strength `σₘₐₓ` and the characteristic normal separation `δₙ`.
"""
xu_needleman_Φₙ(σₘₐₓ, δₙ) = σₘₐₓ * exp(1.0) * δₙ
"""
    xu_needleman_Φₜ(τₘₐₓ, δₜ)

Compute the tangential work of separation `Φₜ` for the Xu-Needleman cohesive law based on the cohesive tangential strength `τₘₐₓ` and the characteristic tangential separation `δₜ`.
"""
xu_needleman_Φₜ(τₘₐₓ, δₜ) = τₘₐₓ * sqrt(0.5exp(1.0)) * δₜ
"""
    xu_needleman_σₘₐₓ(Φₙ, δₙ) = Φₙ / (exp(1.0) * δₙ)

Compute the cohesive normal strength `σₘₐₓ` for the Xu-Needleman cohesive law based on the normal work of separation `Φₙ` and the characteristic normal separation `δₙ`.
"""
xu_needleman_σₘₐₓ(Φₙ, δₙ) = Φₙ / (exp(1.0) * δₙ)
"""
    xu_needleman_τₘₐₓ(Φₜ, δₜ)

Compute the cohesive tangential strength `τₘₐₓ` for the Xu-Needleman cohesive law based on the tangential work of separation `Φₜ` and the characteristic tangential separation `δₜ`.
"""
xu_needleman_τₘₐₓ(Φₜ, δₜ) = Φₜ / (sqrt(0.5exp(1.0)) * δₜ)

struct XuNeedlemanState <: AbstractMaterialState end
initial_material_state(::XuNeedleman) = XuNeedlemanState()

get_cache(::XuNeedleman) = nothing

"""
    material_response(m::XuNeedleman, Δ::Tensor{1,dim}) where dim

Return the traction vector and the traction tangent for the given separation jump `Δ`.
The last entry of `Δ` is interpreted as normal separation, the first entries are interpreted as in-plane separations.

No `MaterialState` is needed for the stress computation, thus if a state is handed over to `material_response`, the same state is returned.
"""
function material_response(m::XuNeedleman, Δ::Tensor{1,dim}, state::XuNeedlemanState=XuNeedlemanState(),
    Δt=nothing, cache=nothing, options=nothing) where dim

    # unpack variables
    Φₜ = m.Φₜ
    Φₙ = m.Φₙ
    δₜ = m.δₜ
    δₙ = m.δₙ
    q = Φₜ/Φₙ
    r = m.Δₙˢ / δₙ

    Φ(Δₜ, Δₙ) = Φₙ + Φₙ*exp(-Δₙ/δₙ) * ((1 - r + Δₙ/δₙ) * (1-q)/(r-1) - (q + (r-q)/(r-1) * Δₙ/δₙ) * exp(-(Δₜ ⋅ Δₜ) / δₜ^2))

    dTdΔ, T, _ = hessian(Δ -> Φ((Tensor{1,dim-1}(i->Δ[i])), last(Δ)), Δ, :all)

    return T, dTdΔ, state
end
