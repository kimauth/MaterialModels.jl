
struct GreenLagrange{T<:SymmetricTensor{2}} <: StrainMeasure; value::T; end
struct RightCauchyGreen{T<:SymmetricTensor{2}} <: StrainMeasure; value::T; end
struct DeformationGradient{T<:Tensor{2}} <: StrainMeasure; value::T; end
struct VelocityGradient{T} <: StrainMeasure; value::T; end
struct SmallStrain{T} <: StrainMeasure; value::T;  end

strainmeasure(m::AbstractMaterial) = error("Strain measure for material $m not defined ")

abstract type StressMeasure end
struct SecondPiolaKirchhoff{T<:SymmetricTensor{2}} <: StressMeasure; value::T; end
struct FirstPiolaKirchhoff{T<:Tensor{2}} <: StressMeasure; value::T; end
struct FirstPiolaKirchhoffTransposed{T<:Tensor{2}} <: StressMeasure; value::T; end
struct TrueStress{T<:SymmetricTensor{2}} <: StressMeasure; value::T; end

abstract type AbstractTangent end
struct ∂S∂C{T<:SymmetricTensor{4}} <: AbstractTangent; value::T; end
struct ∂S∂E{T<:SymmetricTensor{4}} <: AbstractTangent; value::T; end
struct ∂Pᵀ∂F{T<:Tensor{4}} <: AbstractTangent; value::T; end
struct ∂σ∂ε{T<:SymmetricTensor{4}} <: AbstractTangent; value::T; end

"""
    transform_strain(strain::::StrainMeasure, to::Type{StrainMeasure})

Compute strain meassure defined by `to` from `strain`.
"""
transform_strain(strain::Ts, ::Type{Tn}) where {Tn, Ts<:Tn} = strain # generic (no transformation)
function transform_strain(F::DeformationGradient, ::Type{RightCauchyGreen}) 
    C = tdot(F.value)
    return RightCauchyGreen(C)
end
function transform_strain(C::RightCauchyGreen{T}, ::Type{GreenLagrange}) where T
    E = (C.value - one(T))/2
    return GreenLagrange(E)
end
function transform_strain(E::GreenLagrange{T}, ::Type{RightCauchyGreen}) where T
    C = 2E.value + one(T)
    return RightCauchyGreen(C)
end
function transform_strain(F::DeformationGradient, ::Type{GreenLagrange})
    C = tdot(F.value)
    E = (C - one(C))/2
    return GreenLagrange(E)
end

"""
    transform_tangent(stress::StressMeasure, tangent::AbstractTangent, to::Type{AbstractTangent}, strain::StrainMeasure)

Transform the stress and tangent to the stress and tangent types given by the tangent type `to`.
Some transformations require the deformation gradient which can be given by `strain`.
Transformations that do not require the deformation gradient ignore `strain`.
"""
transform_stress_tangent(stress, tangent::T, ::Type{T}, ::StrainMeasure)  where T<:AbstractTangent = stress, tangent
function transform_stress_tangent(S::SecondPiolaKirchhoff, dSdC::∂S∂C, ::Type{∂S∂E}, ::StrainMeasure)
    return S, ∂S∂E(2dSdC.value)
end
function transform_stress_tangent(S::SecondPiolaKirchhoff, dSdE::∂S∂E, ::Type{∂S∂C}, ::StrainMeasure)
    return S, ∂S∂C(0.5dSdE.value)
end
function transform_stress_tangent(
        S::SecondPiolaKirchhoff{T},
        dSdC::∂S∂C, 
        ::Type{∂Pᵀ∂F},
        F::DeformationGradient
) where T
    I = one(T)
    Pᵀ = S.value ⋅ F.value'
    dPᵀdF = otimesl(S.value, I) + otimesu(I, F.value) ⊡ dSdC.value ⊡ (otimesl(I, F.value') + otimesu(F.value', I))
    return FirstPiolaKirchhoff(Pᵀ), ∂Pᵀ∂F(dPᵀdF)
end

"""
    material_response(output_tangent::Type{<:AbstractTangent}, m::AbstractMaterial, strain::StrainMeasure, state, Δt::Float64; cache, options)

Function for automatically converting the output stress and tangent from any material, to any 
other stress/tangent defined by `output_tangent` (dSdC, dPᵀdF etc.) 

!!! note
    Not all stress/strain measures are readily convertible into each other without additional input. 
    E.g. `∂S∂C` and `∂S∂E` can be easily converted without extra inputs, but the conversion
    of `∂S∂C` into `∂Pᵀ∂F` requires the deformation gradient `F` as `input_strain`.
"""
# tangent transformation layer
function material_response(
        ::Type{output_tangent},
        m::M,
        strain::StrainMeasure, # usually DeformationGradient(F)
        state,
        Δt::Float64 = 0.0;
        cache=nothing,
        options=nothing,
    ) where {M<:AbstractMaterial, output_tangent}

    native_strain = transform_strain(strain, native_strain_type(M))
    stress, tangent, newstate = _material_response(m, native_strain, state, Δt; cache=cache, options=options)
    out_stress, out_tangent = transform_stress_tangent(stress, tangent, output_tangent, strain)

    # return tensors (not wrapped stress types)
    return out_stress.value, out_tangent.value, newstate
end

# wrapped stress + tangent types for tangent transformation
function _material_response( # internal as it returns typed stress/tangent
        m::M,
        strain::T,
        state,
        Δt::Float64 = 0.0;
        cache=nothing,
        options=nothing
    ) where {M<:AbstractMaterial, T<:StrainMeasure}

    # TODO: Maybe this check is not needed if this routine is only called from the tangent transformation layer
    compatible_strain_measure(T, native_strain_type(M)) || error("$T is not the native strain measure for $M:")

    # call of native 3D routine
    stress, tangent, new_state = material_response(m, strain.value, state, Δt; cache, options)

    tangent_type = native_tangent_type(M)
    stress_type = _stress_type(tangent_type)

    # return wrapped stress/tangent types for tangent transformations
    return stress_type(stress), tangent_type(tangent), new_state
end
    
# transformation for strain energy
function elastic_strain_energy_density(material::M, strain::StrainMeasure) where M
    native_strain = transform_strain(strain, native_strain_type(M))
    return elastic_strain_energy_density(material, native_strain.value)
end

native_strain_type(::Type{M}) where M<:AbstractMaterial = _strain_type(native_tangent_type(M))
native_stress_type(::Type{M}) where M<:AbstractMaterial = _stress_type(native_tangent_type(M))
_strain_type(::Type{∂S∂C}) = RightCauchyGreen
_stress_type(::Type{∂S∂C}) = SecondPiolaKirchhoff
_strain_type(::Type{∂σ∂ε}) = SmallStrain
_stress_type(::Type{∂σ∂ε}) = TrueStress

# same strain type is compatible
compatible_strain_measure(::Type{<:T}, ::Type{T}) where T = true
# different strain type is not compatible
compatible_strain_measure(::Type{<:T1}, ::Type{T2}) where {T1, T2} = false
