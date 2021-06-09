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

function vector_residual!(R::Function, r_vector::AbstractVector{T}, x_vector::AbstractVector{T}, m) where T
    # construct residuals with type T
    x_tensor = frommandel(Tensors.get_base(typeof(m)), x_vector)
    r_tensor = R(x_tensor)
    tomandel!(r_vector, r_tensor)
    return r_vector
end

function update_cache!(cache, f)
    cache.f = f
    chunk = ForwardDiff.Chunk(cache.x_f)
    jac_cfg = ForwardDiff.JacobianConfig(cache.f, cache.F, cache.x_f, chunk)
    ForwardDiff.checktag(jac_cfg, cache.f, cache.x_f)

    F2 = copy(cache.F)
    function j_forwarddiff!(J, x)
        ForwardDiff.jacobian!(J, cache.f, F2, x, jac_cfg, Val{false}())
    end
    function fj_forwarddiff!(F, J, x)
        jac_res = DiffResults.DiffResult(F, J)
        ForwardDiff.jacobian!(jac_res, cache.f, F2, x, jac_cfg, Val{false}())
        DiffResults.value(jac_res)
    end

    cache.df = j_forwarddiff!
    cache.fdf = fj_forwarddiff!
    return cache
end