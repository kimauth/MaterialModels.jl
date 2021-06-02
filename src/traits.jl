struct SmallStrain <: StrainMeasure end
struct GreenLagrangeStrain <: StrainMeasure end
struct RightCauchyGreen <: StrainMeasure end
struct DeformationGradient <: StrainMeasure end
struct VelocityGradient <: StrainMeasure end

strainmeasure(::Type{<:AbstractMaterial}) = error("Strain measure for material $m not defined ")

#
# Main dispatch
function material_response(m::M, F::Tensor{2}, state::AbstractMaterialState, Δt::Float64, cache, options) where M<:AbstractMaterial 
     constitutive_driver(strainmeasure(M), m, F, state, Δt, cache, options)
end

#
# RightCauchyGreen
function material_response(::RightCauchyGreen, m::AbstractMaterial, F, state::AbstractMaterialState, Δt::Float64, cache, options)
    C = tdot(F)
    S, dSdE, state = constitutive_driver(m, C, state::AbstractMaterialState, Δt::Float64, cache, options)
end

#
# DeformationGradient
function material_response(::DeformationGradient, m::AbstractMaterial, F::Tensor{2}, state::AbstractMaterialState, Δt::Float64, cache, options)
    S, dSdE, state = constitutive_driver(m, F, state::AbstractMaterialState, Δt::Float64, cache, options)
end

#
# SmallStrain
#function material_response(::DeformationGradientStrain, m::AbstractMaterial, F::Tensor{2}, state::AbstractMaterialState, Δt::Float64, cache, options)
#    ε = symmetric(F) - one(F)
#    σ, dσdε, state = constitutive_driver(m, F, state::AbstractMaterialState, Δt::Float64, cache, options)
#end
