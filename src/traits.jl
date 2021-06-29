export dSdC, dSdE, dPᵀdF

struct GreenLagrange <: StrainMeasure end
struct RightCauchyGreen <: StrainMeasure end
struct DeformationGradient <: StrainMeasure end
struct VelocityGradient <: StrainMeasure end

strainmeasure(::Type{<:AbstractMaterial}) = error("Strain measure for material $m not defined ")

abstract type AbstractTangent end
struct dSdC <: AbstractTangent end
struct dSdE <: AbstractTangent end
struct dPᵀdF <: AbstractTangent end

#Here we must associate a strain meassure with an output tangent
# These default makes sence: 
default_tangent(::GreenLagrange) = dSdE()
default_tangent(::RightCauchyGreen) = dSdC()
default_tangent(::DeformationGradient) = dPᵀdF()

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
transform_tangent(S, dSdC, F::Tensor{2}, from::dSdC, to::dSdE) = S, 2dSdC
transform_tangent(S, dSdC, F::Tensor{2}, from::dSdC, to::dPᵀdF) = S⋅F', otimesu(F,I) ⊡ dSdC ⊡ otimesu(F',I) + otimesu(S,I)
transform_tangent(S, dSdC, F::Tensor{2}, from::dPᵀdF, to::dSdC) = begin
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
    return material_response(output_tangent, strainmeasure(m), m, F, state, Δt, cache=cache, options=options)
end

function material_response(output_tangent::AbstractTangent, straintype::StrainMeasure, m::AbstractMaterial, F::Tensor{2}, state::AbstractMaterialState, Δt::Float64; cache, options)
    
    strain = compute_strain(F, straintype)
    stress, strain, newstate = material_response(m, strain, state, Δt, cache=cache, options=options)
    out_stress, out_tangent = transform_tangent(stress, strain, F, default_tangent(straintype), output_tangent)

    return out_stress, out_tangent, newstate
end
