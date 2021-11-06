
"""
    XuNeedleman(σₘₐₓ, τₘₐₓ, δₙ, δₜ, Φₙ, Φₜ, Δₙˢ)

Xu-Needleman traction-separation law.[^Xu1993] This is a commonly used cohesive law for brittle fracture, 
however it is revertible and thus not suited for unloading. In 3D, it is isotropic within the cohesive plane.

[^Xu1993]: Xu, X. P., & Needleman, A. (1993). Void nucleation by inclusion debonding in a crystal matrix. Modelling and Simulation in Materials Science and Engineering, 1(2), 111–132. [https://doi.org/10.1088/0965-0393/1/2/001](https://doi.org/10.1088/0965-0393/1/2/001)

# Arguments
- `σₘₐₓ::Float64`: cohesive normal strength
- `τₘₐₓ::Float64`: cohesive tangential strength (in-plane strength)
- `δₙ::Float64`: characteristic normal separation
- `δₜ`::Float64`: characteristic tangential separation
- `Φₙ::Float64`: normal work of separation
- `Φₜ::Float64`: tangential work of separation
- `Δₙˢ`: normal separation after complete shear separation under the condition of zero normal tension

Out of the pairs cohesive strength / characteristic normal separation / work of separation any two can be given
for construction the `XuNeedleman` material.
"""
struct XuNeedleman <: AbstractMaterial
    σₘₐₓ::Float64
    τₘₐₓ::Float64
    δₙ::Float64
    δₜ::Float64
    Φₙ::Float64
    Φₜ::Float64
    Δₙˢ::Float64
end

function XuNeedleman(;
    σₘₐₓ::Union{Nothing, Float64}=nothing,
    τₘₐₓ::Union{Nothing, Float64}=nothing,
    δₙ::Union{Nothing, Float64}=nothing,
    δₜ::Union{Nothing, Float64}=nothing,
    Φₙ::Union{Nothing, Float64}=nothing,
    Φₜ::Union{Nothing, Float64}=nothing,
    Δₙˢ::Float64,
)
    p_strength = σₘₐₓ !== nothing && τₘₐₓ !== nothing
    p_jump = δₙ !== nothing && δₜ !== nothing
    p_work = Φₙ !== nothing && Φₜ !== nothing
    p_strength + p_jump + p_work == 2 || error("You need to prescribe exactly 2 pairs out of cohesive strength, characteristic separations and work of separation.")

    if !p_strength 
        σₘₐₓ = Φₙ / (exp(1.0) * δₙ)
        τₘₐₓ = Φₜ / (sqrt(0.5exp(1.0)) * δₜ)
    elseif !p_jump
        δₙ = Φₙ / (exp(1.0) * σₘₐₓ)
        δₜ = Φₜ / (sqrt(0.5exp(1.0)) * τₘₐₓ)
    elseif !p_work
        Φₙ = σₘₐₓ * exp(1.0) * δₙ
        Φₜ = τₘₐₓ * sqrt(0.5exp(1.0)) * δₜ
    end

    return XuNeedleman(σₘₐₓ, τₘₐₓ, δₙ, δₜ, Φₙ, Φₜ, Δₙˢ)
end

struct XuNeedlemanState <: AbstractMaterialState end
initial_material_state(::XuNeedleman) = XuNeedlemanState()

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

    dTdΔ, T, _ = hessian(Δ -> Φ((@view Δ[1:end-1]), last(Δ)), Δ, :all)

    return T, dTdΔ, state
end