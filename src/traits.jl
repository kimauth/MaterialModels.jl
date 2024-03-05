# external: Wrappers for strain (user interface of material_response)
struct GreenLagrange{T<:SymmetricTensor{2}} <: StrainMeasure; value::T; end
struct RightCauchyGreen{T<:SymmetricTensor{2}} <: StrainMeasure; value::T; end
struct DeformationGradient{T<:Tensor{2}} <: StrainMeasure; value::T; end
struct VelocityGradient{T} <: StrainMeasure; value::T; end
struct SmallStrain{T} <: StrainMeasure; value::T;  end
# allow to index strain wrappers like normal tensors
Base.getindex(A::StrainMeasure, idxs::Int...) = Base.getindex(A.value, idxs...)

get_base(::Type{<:DeformationGradient}) = DeformationGradient
get_base(::Type{<:RightCauchyGreen}) = RightCauchyGreen
get_base(::Type{<:GreenLagrange}) = GreenLagrange
get_base(::Type{<:VelocityGradient}) = VelocityGradient
get_base(::Type{<:SmallStrain}) = SmallStrain

# allow get_base perform as unity on tensors
get_base(::Type{T}) where T<:AbstractTensor = T

increase_dim(strain::SM, neutral) where SM<:StrainMeasure = get_base(SM)(increase_dim(strain.value, neutral))

strainmeasure(m::AbstractMaterial) = error("Strain measure for material $m not defined ")

# external: User interface to define which measures to return
abstract type AbstractTangent end
struct ∂S∂C <: AbstractTangent; end
struct ∂S∂E <: AbstractTangent; end
struct ∂P∂F <: AbstractTangent; end
struct ∂Pᵀ∂F <: AbstractTangent; end
struct ∂σ∂ε <: AbstractTangent; end

# internal: singleton types for trait system
struct _GreenLagrange <: StrainMeasure; end
struct _RightCauchyGreen <: StrainMeasure; end
struct _DeformationGradient <: StrainMeasure; end
struct _VelocityGradient <: StrainMeasure; end
struct _SmallStrain <: StrainMeasure; end

abstract type StressMeasure end
struct _SecondPiolaKirchhoff <: StressMeasure; end
struct _FirstPiolaKirchhoff <: StressMeasure; end
struct _FirstPiolaKirchhoffTransposed <: StressMeasure; end
struct _TrueStress <: StressMeasure; end

# assign stress/strain traits to tangents
straintrait(::Type{∂S∂C}) = _RightCauchyGreen()
straintrait(::Type{∂S∂E}) = _GreenLagrange()
straintrait(::Type{∂P∂F}) = _DeformationGradient()
straintrait(::Type{∂Pᵀ∂F}) = _DeformationGradient()
straintrait(::Type{∂σ∂ε}) = _SmallStrain()

stresstrait(::Type{∂S∂C}) = _SecondPiolaKirchhoff()
stresstrait(::Type{∂S∂E}) = _SecondPiolaKirchhoff()
stresstrait(::Type{∂P∂F}) = _FirstPiolaKirchhoff()
stresstrait(::Type{∂Pᵀ∂F}) = _FirstPiolaKirchhoffTransposed()
stresstrait(::Type{∂σ∂ε}) = _TrueStress()

# assign strain wrappers to strain trait types
straintrait(::Type{<:RightCauchyGreen}) = _RightCauchyGreen()
straintrait(::Type{<:GreenLagrange}) = _GreenLagrange()
straintrait(::Type{<:DeformationGradient}) = _DeformationGradient()
straintrait(::Type{<:SmallStrain}) = _SmallStrain()
"""
    transform_strain(strain::StrainMeasure, to::Type{StrainMeasure})

Compute strain meassure defined by `to` from `strain`.
"""
function transform_strain(strain::T1, ::T2) where {T1, T2} # generic (no transformation)
    if typeof(straintrait(T1)) != T2
        error("The strain conversion from $T1 to $T2 is not implemented.")
    end
    return strain
end
function transform_strain(F::DeformationGradient, ::_RightCauchyGreen) 
    C = tdot(F.value)
    return RightCauchyGreen(C)
end
function transform_strain(C::RightCauchyGreen{T}, ::_GreenLagrange) where T
    E = (C.value - one(T))/2
    return GreenLagrange(E)
end
function transform_strain(E::GreenLagrange{T}, ::_RightCauchyGreen) where T
    C = 2E.value + one(T)
    return RightCauchyGreen{T}(C)
end
function transform_strain(F::DeformationGradient, ::_GreenLagrange)
    C = tdot(F.value)
    E = (C - one(C))/2
    return GreenLagrange(E)
end
# forward calls to the traits
# TODO: perhaps not even needed?
transform_strain(strain::StrainMeasure, ::T) where T<:AbstractTangent = transform_strain(strain, straintrait(T))

"""
    transform_tangent(stress::StressMeasure, tangent::AbstractTangent, to::Type{AbstractTangent}, strain::StrainMeasure)

Transform the stress and tangent to the stress and tangent types given by the tangent type `to`.
Some transformations require the deformation gradient which can be given by `strain`.
Transformations that do not require the deformation gradient ignore `strain`.
"""
transform_stress_tangent(stress::SecondOrderTensor, tangent::FourthOrderTensor, 
        #=from=#::T, #=to=#::T, ::StrainMeasure) where T<:AbstractTangent = stress, tangent

transform_stress_tangent(S::SymmetricTensor{2}, dSdC::SymmetricTensor{4}, 
        #=from=#::∂S∂C, #=to=#::∂S∂E, ::StrainMeasure) = S, 2dSdC

transform_stress_tangent(S::SymmetricTensor{2}, dSdE::SymmetricTensor{4}, 
        #=from=#::∂S∂E, #=to=#::∂S∂C, ::StrainMeasure) = S, 0.5dSdE

function transform_stress_tangent(S::T, dSdC::SymmetricTensor{4}, 
        #=from=#::∂S∂C, #=to=#::∂Pᵀ∂F, F::DeformationGradient) where T<:SymmetricTensor{2}
    I = one(T)
    Pᵀ = S ⋅ F.value'
    dPᵀdF = otimesl(S, I) + otimesu(I, F.value) ⊡ dSdC ⊡ (otimesl(I, F.value') + otimesu(F.value', I))
    return Pᵀ, dPᵀdF
end

function transform_stress_tangent(S::SymmetricTensor{2,dim,T}, dSdC::SymmetricTensor{4}, 
        #=from=#::∂S∂C, #=to=#::∂P∂F, F::DeformationGradient) where {dim, T}
    P = F.value ⋅ S
    dPdF = Tensor{4,dim,T}() do i,j,k,l
        v = i == k ? S[l,j] : zero(T)
        for n in 1:dim, m in 1:dim
            v += F[i,m] * (dSdC[m,j,l,n] * F[k,n] + dSdC[m,j,n,l] * F[k,n])
        end
        return v
    end
    return P, dPdF
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
        output_tangent::AbstractTangent,
        m::M,
        strain::StrainMeasure, # usually DeformationGradient(F)
        args...;
        varargs...
    ) where M

    native_strain = transform_strain(strain, native_tangent(M))
    stress, tangent, other_returns... = material_response(m, native_strain.value, args...; varargs...)
    out_stress, out_tangent = transform_stress_tangent(stress, tangent, native_tangent(M), output_tangent, strain)

    return out_stress, out_tangent, other_returns...
end
   
# transformation for strain energy
function elastic_strain_energy_density(material::M, strain::StrainMeasure) where M
    native_strain = transform_strain(strain, native_tangent(M))
    return elastic_strain_energy_density(material, native_strain.value)
end
