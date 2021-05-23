# # generic fallback, probably slow. TODO: can we have a fast version of this?
# # Is a macro more suitable here?
# function Tensors.tomandel!(v::Vector{T}, r::Residuals) where T
#     # TODO check for length of v
#     start = 1
#     for fn in fieldnames(typeof(r))
#         start = _set_mandel_values!(v, getfield(r, fn), start)
#     end
#     return v
# end

# function _set_mandel_values!(v::Vector{T}, r::SymmetricTensor{order, dim, T, M}, start::Int) where {order, dim, T, M}
#     tomandel!(v[start:end], r)
#     return start + M
# end
# function _set_mandel_values!(v::Vector{T}, r::T, start::Int) where T
#     v[start] = r
#     return start + 1
# end

function vector_residual!(R::Function, r_vector::Vector{T}, x_vector::Vector{T}, m::AbstractMaterial) where T
    # construct residuals with type T
    x_tensor = frommandel(Residuals{typeof(m)}, x_vector)
    r_tensor = R(x_tensor)
    tomandel!(r_vector, r_tensor)
    return r_vector
end