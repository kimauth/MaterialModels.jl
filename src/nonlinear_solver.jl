##############################################
# NLsolve.jl related functionality
##############################################

function vector_residual!(R::Function, r_vector::Vector{T}, x_vector::Vector{T}, m) where T
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

##############################################
# Newton solver for isbits residuals
##############################################

"""
    solve(f::Function, x::Union{Real, SVector}; max_iter=100, tol=1e-8)

Find a solution `x` such that `f(x) = 0` by Newton-Raphson iterations.
Returns `Tuple(x, dfdx)` upon convergence, else returns `nothing`.

Keyword arguments:
- `max_iter`: Maximum number of iterations.
- `tol`: Tolerance for convergence. The solution is converged if `norm(f(x)) < tol`.

!!! info
    Currently only `SVector` and scalar type residuals are solvable. It could be extended to
    `Tensor` residuals in a straight forward manner. 
"""
function solve(f::Function, x::Union{Real, SVector}; max_iter=100, tol=1e-8)
    for i = 1:max_iter
        r = f(x)
        drdx = auto_diff(f, x)
        if norm(r) < tol
            return (x, drdx)
        end
        x -= drdx \ r
    end
    return nothing
end

auto_diff(f::Function, x::Real) = ForwardDiff.derivative(f, x)
auto_diff(f::Function, x::SVector) = ForwardDiff.jacobian(f, x)