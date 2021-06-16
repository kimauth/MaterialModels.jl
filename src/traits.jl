
struct GreenLagrange <: StrainMeasure end
struct RightCauchyGreen <: StrainMeasure end
struct DeformationGradient <: StrainMeasure end
struct VelocityGradient <: StrainMeasure end

strainmeasure(::Type{<:AbstractMaterial}) = error("Strain measure for material $m not defined ")

"""
    material_response_S_dSdE(m::AbstractMaterial, F::Tensor{2}, state::AbstractMaterialState, Δt::Float64; cache, options)

Function for automatically converting the stress and tangent from any material model to S and dSdE 
"""
function material_response_S_dSdE(m::M, F::Tensor{2}, state::AbstractMaterialState, Δt::Float64; cache, options) where M<:AbstractMaterial 
    material_response_S_dSdE(strainmeasure(M), m, F, state, Δt, cahce=cache, options=options)
end

#
# RightCauchyGreen
function material_response_S_dSdE(::RightCauchyGreen, m::AbstractMaterial, F::Tensor{2}, state::AbstractMaterialState, Δt::Float64; cache, options)
    C = tdot(F)
    S, dSdC, state = material_response(m, C, stat, Δt, cahce=cache, options=options)
    return S, 2dSdC, state
end

#
# GreenLagrange
function material_response_S_dSdE(::GreenLagrange, m::AbstractMaterial, F::Tensor{2}, state::AbstractMaterialState, Δt::Float64; cache, options)
    E = symmetric(1/2 * (F' ⋅ F - one(F)))
    S, dSdE, state = material_response(m, E, state, Δt, cahce=cache, options=options)
    return S, dSdE, state
end

#
# DeformationGradient
function material_response_S_dSdE(::DeformationGradient, m::AbstractMaterial, F::Tensor{2}, state::AbstractMaterialState, Δt::Float64; cache, options)
    Pᵀ, dPᵀdF, state = material_response(m, F, state, Δt, cahce=cache, options=options)
    S = Pᵀ ⋅ inv(F')
    dSdE = 2 * inv( otimesu(F,I) ) ⊡ (dPᵀdF - otimesu(S,I)) ⊡ inv( otimesu(F',I) )
    return S, 2dSdC, state 
end

"""
    material_response_Pᵀ_dPᵀdF(m::AbstractMaterial, F::Tensor{2}, state::AbstractMaterialState, Δt::Float64; cache, options)

Function for automatically converting the stress and tangent from any material model to Pᵀ and dPᵀdF 
"""
function material_response_Pᵀ_dPᵀdF(m::M, F::Tensor{2}, state::AbstractMaterialState, Δt::Float64; cache, options) where M<:AbstractMaterial 
    material_response_Pᵀ_dPᵀdF(strainmeasure(M), m, F, state, Δt, cache, options)
end

#
# DeformationGradient
function material_response_Pᵀ_dPᵀdF(::DeformationGradient, m::AbstractMaterial, F::Tensor{2}, state::AbstractMaterialState, Δt::Float64; cache, options)
    Pᵀ, dPᵀdF, state = material_response(m, F, state, Δt, cahce=cache, options=options)
    return Pᵀ, dPᵀdF, state
end

#
# RightCauchyGreen
function material_response_Pᵀ_dPᵀdF(::RightCauchyGreen, m::AbstractMaterial, F::Tensor{2}, state::AbstractMaterialState, Δt::Float64; cache, options)
    C = tdot(F)
    S, dSdC, state = material_response(m, C, state, Δt, cahce=cache, options=options)
    dPᵀdF =  otimesu(F,I) ⊡ dSdE ⊡ otimesu(F',I) + otimesu(S,I)
    Pᵀ = S⋅F'
    return Pᵀ, dPᵀdF, state
end

#
# GreenLagrange
# ...and so on
