abstract type AbstractFiniteStrainMaterial <: AbstractMaterial end

# increase dimension of strain tensors
neutral_element(::AbstractFiniteStrainMaterial) = one
