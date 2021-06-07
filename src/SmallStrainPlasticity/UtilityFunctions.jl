# Should be fixed in ForwardDiff?
DiffResults.DiffResult(value::MArray, derivs::Tuple{Vararg{MArray}}) = DiffResults.MutableDiffResult(value, derivs)

# Generic functions, should be defined elsewhere?
function vonMises(𝛔::SymmetricTensor{2,3})
    𝛔_dev = dev(𝛔)
    return sqrt((3.0/2.0) * (𝛔_dev ⊡ 𝛔_dev))
end

function vonMisesDev(𝛔_dev::SymmetricTensor{2,3})
    return sqrt((3.0/2.0) * (𝛔_dev ⊡ 𝛔_dev))
end

macaulay(x::T) where {T} = x > 0.0 ? x : zero(T)