
struct GreenLagrange <: StrainMeasure end
struct RightCauchyGreen <: StrainMeasure end
struct DeformationGradient <: StrainMeasure end
struct VelocityGradient <: StrainMeasure end
struct EngineeringStrain <: StrainMeasure end

strainmeasure(m::AbstractMaterial) = error("Strain measure for material $m not defined ")

abstract type AbstractTangent end
struct ∂S∂C <: AbstractTangent end
struct ∂S∂E <: AbstractTangent end
struct ∂Pᵀ∂F <: AbstractTangent end

#Here we must associate a strain meassure with an output tangent
# These default makes sence: 
default_tangent(::GreenLagrange) = ∂S∂E()
default_tangent(::RightCauchyGreen) = ∂S∂C()
default_tangent(::DeformationGradient) = ∂Pᵀ∂F()

"""
    compute_strain(F::Tensor{2}, stype::StrainMeasure)

Compute strain meassure defined by `stype`, using the deformation gradient `F`
"""
compute_strain(F::Tensor{2}, ::RightCauchyGreen) = tdot(F)
compute_strain(F::Tensor{2}, ::GreenLagrange) = symmetric(1/2 * (F' ⋅ F - one(F)))
compute_strain(F::Tensor{2}, ::DeformationGradient) = F

"""
    transform_tangent(stress, tangent, F::Tensor{2}, from::AbstractTangent, to::AbstractTangent)

Transform the stress and tangent
"""
transform_tangent(S , dSdC,  F::Tensor{2}, from::∂S∂C, to::∂S∂C) = S, dSdC
transform_tangent(Pᵀ, ∂Pᵀ∂F, F::Tensor{2}, from::∂Pᵀ∂F, to::∂Pᵀ∂F) = Pᵀ, ∂Pᵀ∂F
transform_tangent(S , ∂S∂E,  F::Tensor{2}, from::∂S∂E, to::∂S∂E) = S, ∂S∂E

transform_tangent(S, dSdC, F::Tensor{2}, from::∂S∂C, to::∂S∂E) = S, 2dSdC
transform_tangent(S, dSdC, F::Tensor{2,3,T}, from::∂S∂C, to::∂Pᵀ∂F) where T = begin 
    I = one(SymmetricTensor{2,3,T})
    Pᵀ = S⋅F'
    dPᵀdF = otimesu(F,I) ⊡ dSdC ⊡ otimesu(F',I) + otimesu(S,I)
    return Pᵀ, dPᵀdF
end
transform_tangent(Pᵀ, dPᵀdF, F::Tensor{2}, from::∂Pᵀ∂F, to::∂S∂C) = begin
    I = one(SymmetricTensor{2,3,T})
    S = Pᵀ ⋅ inv(F')
    dSdC = 2 * inv( otimesu(F,I) ) ⊡ (dPᵀdF - otimesu(S,I)) ⊡ inv( otimesu(F',I) )
    return S, dSdC
end
#...

"""
    material_response(output_tangent::AbstractTangent, m::AbstractMaterial, F::Tensor{2}, state::AbstractMaterialState, Δt::Float64; cache, options)

Function for automatically converting the output stress and tangent from any material, to any 
other stress/tangent defined by `output_tangent` (dSdC, dPᵀdF etc.) 
"""
function material_response(output_tangent::AbstractTangent, m::AbstractMaterial, F::Tensor{2}, state::AbstractMaterialState, Δt::Float64 = 0.0; cache=nothing, options=nothing)
    material_response(ThreeD(), output_tangent, m, F, state, Δt; cache=cache, options=options)
end

function material_response(dim::AbstractDim, output_tangent::AbstractTangent, m::AbstractMaterial, F::Tensor{2}, state::AbstractMaterialState, Δt::Float64 = 0.0; cache=nothing, options=nothing)
    straintype = strainmeasure(m);
    strain = compute_strain(F, straintype)
    stress, strain, newstate = material_response(dim, m, strain, state, Δt, cache=cache, options=options)
    out_stress, out_tangent = transform_tangent(stress, strain, F, default_tangent(straintype), output_tangent)

    return out_stress, out_tangent, newstate
end