using Tensors
using NLsolve

mutable struct Residuals{T}
    σ::SymmetricTensor{2,3,T,6}
    κ::T
    α::SymmetricTensor{2,3,T,6}
    μ::T
end

function frommandel!(r::Residuals{T}, v::Vector{T}) where T
    M=6
    r.σ = frommandel(SymmetricTensor{2,3}, view(v, 1:M))
    r.κ = v[M+1]
    r.α = frommandel(SymmetricTensor{2,3}, view(v, M+2:2M+1))
    r.μ = v[2M+2]
    return r
end

function foo_nocrash(cache)
    x = Residuals(ones(SymmetricTensor{2,3}), 1.0, ones(SymmetricTensor{2,3}), 1.0)
    res = nlsolve(cache, ones(14); method=:newton)

    frommandel!(x, res.zero)
end

function foo_crash(cache, options)
    x = Residuals(ones(SymmetricTensor{2,3}), 1.0, ones(SymmetricTensor{2,3}), 1.0)
    res = nlsolve(cache, ones(14); options...)
    
    frommandel!(x, res.zero)
end

A = rand(14, 14)
function f!(b, x)
    b[:] = A*x
end
cache = NLsolve.OnceDifferentiable(f!, rand(14), rand(14); autodiff=:forward)
options = Dict{Symbol, Any}(:method=>:newton)

# no problem if kwargs are given explicitely
foo_nocrash(cache)

# kwargs in Dict and call from function crashes
foo_crash(cache, options)