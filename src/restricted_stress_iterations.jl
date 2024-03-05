#=
function material_response(
    dim::Union{UniaxialStress{d}, PlaneStress{d}},
    output_tangent::AbstractTangent,
    m::AbstractMaterial,
    ε::StrainMeasure,
    state = nothing,
    Δt = nothing;
    cache =  get_cache(m),
    options = NamedTuple(), 
    ) where d
    
    # iterations need to happen with the strain measure corresponding to the output tangent
    strain = transform_strain(ε, output_tangent)
    return _stress_iterations(dim, (output_tangent, m), strain, (), state, Δt; cache, options)
end


function material_response(
    dim::Union{UniaxialStress{d}, PlaneStress{d}},
    m::AbstractMaterial,
    ε::SecondOrderTensor{d},
    state = nothing,
    Δt = nothing;
    cache =  get_cache(m),
    options = NamedTuple(), 
    ) where d
    
    return _stress_iterations(dim, (m,), ε, (), state, Δt; cache, options)
end
=#



# out of plane / axis components for restricted stress cases
get_zero_indices(::PlaneStress{2}, ::SymmetricTensor{2,3}) = @SVector [3, 4, 5] # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::PlaneStress{2}, ::SymmetricTensor{2,3}) = @SVector [1, 2, 6] # for voigt/mandel format, do not use on tensor data!
get_zero_indices(::PlaneStress{2}, ::Tensor{2,3}) = @SVector [3, 4, 5, 7, 8] # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::PlaneStress{2}, ::Tensor{2,3}) = @SVector [1, 2, 6, 9] # for voigt/mandel format, do not use on tensor data!

get_zero_indices(::UniaxialStress{1}, ::SymmetricTensor{2,3}) = @SVector [i for i in 2:6] # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::UniaxialStress{1}, ::SymmetricTensor{2,3}) = @SVector [1] # for voigt/mandel format, do not use on tensor data!
get_zero_indices(::UniaxialStress{1}, ::Tensor{2,3}) = @SVector [i for i in 2:9] # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::UniaxialStress{1}, ::Tensor{2,3}) = @SVector [1] # for voigt/mandel format, do not use on tensor data!

# finite strain measures
get_zero_indices(dim::AbstractDim, strain::StrainMeasure) = get_zero_indices(dim, strain.value)
get_nonzero_indices(dim::AbstractDim, strain::StrainMeasure) = get_nonzero_indices(dim, strain.value)
# really no good practice
Tensors.tovoigt(T, strain::StrainMeasure) = tovoigt(T, strain.value)
_strain_frommandel(strain::StrainMeasure, v) = fromvoigt(typeof(strain.value), v)
_strain_frommandel(::T, v) where T<:AbstractTensor = fromvoigt(T, v)
