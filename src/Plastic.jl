
"""
    Plastic(E, ν, σ_y, H, r, κ_∞, α_∞)

Plasticity with von Mises yield surface and mixed non-linear kinematic + non-linear isotropic hardening. Both hardening laws are of saturation type.
# Arguments
- `E::Float64`: Young's modulus
- `ν::Float64`: Poisson's ratio
- `σ_y:` yield limit
- `H:` hardening modulus
- `r:` coupling parameter between isotropic and kinematic hardening
- `κ_∞:` saturation stress for isotropic hardening
- `α_∞:` saturation stress for kinematic hardening
"""
struct Plastic <: AbstractMaterial
    E::Float64 # Young's modulus
    ν::Float64 # Poisson's ratio
    σ_y::Float64 # yield limit
    H::Float64 # hardening modulus
    r::Float64 # coupling parameter between isotropic and kinmatic hardening
    κ_∞::Float64 # saturation stress for isotropic hardening
    α_∞::Float64 # saturation stress for kinematic hardening

    # precomputed elastic tangent tensor and hardening moduli
    Eᵉ::SymmetricTensor{4,3,Float64,36}
    H_κ::Float64
    H_α::Float64

    function Plastic(E, ν, σ_y, H, r, κ_∞, α_∞)
        Eᵉ = elastic_tangent_3D(E, ν)
        H_κ = -r*H
        H_α = -2/3*(1-r)*H

        return new(E, ν, σ_y, H, r, κ_∞, α_∞, Eᵉ, H_κ, H_α)
    end
end

# keyword argument constructor
Plastic(;E, ν, σ_y, H, r, κ_∞, α_∞) = Plastic(E, ν, σ_y, H, r, κ_∞, α_∞)

struct PlasticState{dim,T,M} <: AbstractMaterialState
    εᵖ::SymmetricTensor{2,dim,T,M}
    κ::T
    α::SymmetricTensor{2,dim,T,M}
    μ::T
end

Base.zero(::Type{PlasticState{dim,T,M}}) where {dim,T,M} = PlasticState(zero(SymmetricTensor{2,dim,T,M}), zero(T), zero(SymmetricTensor{2,dim,T,M}), zero(T))
initial_material_state(::Plastic) = zero(PlasticState{3,Float64,6})

function get_cache(m::Plastic)
    state = initial_material_state(m)
    # it doesn't actually matter which state and strain step we use here,
    # f is overwritten in the constitutive driver before being used.
    f(r_vector, x_vector) = vector_residual!(((x)->MaterialModels.residuals(x, m, state, zero(SymmetricTensor{2,3}))), r_vector, x_vector, m)
    v_cache = Vector{Float64}(undef, get_n_scalar_equations(m))
    cache = NLsolve.OnceDifferentiable(f, v_cache, v_cache; autodiff = :forward)
    return cache
end

# constitutive driver operates in 3D, so these can always be 3D
# TODO: Should Residuals have a Type Parameter N for the number of scalar equations?
struct ResidualsPlastic{T}
    εᵖ::SymmetricTensor{2,3,T,6}
    κ::T
    α::SymmetricTensor{2,3,T,6}
    μ::T
end

# doesn't allow other dimesion or unsymmetric tensors, so this is a constant
get_n_scalar_equations(::Plastic) = 14

# Residuals(::Plastic) = ResidualsPlastic(zero(SymmetricTensor{2,3}), zero(Float64), zero(SymmetricTensor{2,3}), zero(Float64))
Tensors.get_base(::Type{Plastic}) = ResidualsPlastic # needed for frommandel

# specified version, fewer allocations, faster than generic fallback
function Tensors.tomandel!(v::Vector{T}, r::ResidualsPlastic{T}) where T
    M=6
    # TODO check vector length
    tomandel!(view(v, 1:M), r.εᵖ)
    v[M+1] = r.κ
    tomandel!(view(v, M+2:2M+1), r.α)
    v[2M+2] = r.μ
    return v
end

function Tensors.frommandel(::Type{ResidualsPlastic}, v::Vector{T}) where T
    εᵖ = frommandel(SymmetricTensor{2,3}, view(v, 1:6))
    κ = v[7]
    α = frommandel(SymmetricTensor{2,3}, view(v, 8:13))
    μ = v[14]
    return ResidualsPlastic{T}(εᵖ, κ, α, μ)
end

"""
    material_response(m::Plastic, Δε::SymmetricTensor{2,3,T,6}, state::PlasticState{3}; <keyword arguments>)

Return the stress tensor, stress tangent and the new `MaterialState` for the given strain step Δε and previous material state `state`.

Plastic free energy:
```math
\\Psi^\\text{p} = \\frac{1}{2} \\, r \\, H \\, k^2
+ \\frac{1}{2}\\left( 1-r \\right) \\, H \\, \\left[ \\sqrt{\\frac{2}{3}} \\left| \\text{dev} \\left(\\mathbf{a} \\right) \\right| \\right]^2
```
Von Mises yield function:
```math
\\Phi = \\sqrt{\\frac{3}{2}} \\left| \\text{dev} \\left( \\boldsymbol{\\sigma} - \\boldsymbol{\\alpha} \\right) \\right| - \\sigma_y - \\kappa
```
An associative flow rule and non-associative hardening rules are used. The evolution equations for the hardening variables are:
```math
\\begin{aligned}
\\dot{k} &= -\\lambda \\left( 1 - \\frac{\\kappa}{\\kappa_{\\infty}} \\right) \\\\
\\dot{\\mathbf{a}} &= -\\lambda \\left( \\frac{\\partial\\Phi}{\\partial\\boldsymbol{\\sigma}} 
+ \\frac{3}{2\\alpha_{\\infty}} \\, \\text{dev} \\left( \\boldsymbol{\\alpha} \\right) \\right) \\,.
\\end{aligned}
```
# Keyword arguments
- `cache`: Cache for the iterative solver, used by NLsolve.jl. It is strongly recommended to pre-allocate the cache for repeated calls to `material_response`. See [`get_cache`](@ref).
- `options::Dict{Symbol, Any}`: Solver options for the non-linear solver. Under the key `:nlsolve_params` keyword arguments for `nlsolve` can be handed over.
See [NLsolve documentation](https://github.com/JuliaNLSolvers/NLsolve.jl#common-options). By default the Newton solver will be used.
"""
function material_response(m::Plastic, ε::SymmetricTensor{2,3,T,6}, state::PlasticState{3},
    Δt=nothing; cache=get_cache(m), options::Dict{Symbol, Any} = Dict{Symbol, Any}()) where T

    σ_trial = m.Eᵉ ⊡ (ε - state.εᵖ)
    Φ = sqrt(3/2)*norm(dev(σ_trial-state.α)) - m.σ_y - state.κ

    if Φ <= 0
        return σ_trial, m.Eᵉ, PlasticState(σ_trial, state.κ, state.α, state.μ)
    else
        # set the current residual function that depends only on the variables
        # cache.f = (r_vector, x_vector) -> vector_residual!(((x)->residuals(x,m,state,Δε)), r_vector, x_vector, m)
        f(r_vector, x_vector) = vector_residual!(((x)->residuals(x,m,state,ε)), r_vector, x_vector, m)
        update_cache!(cache, f)
        # initial guess
        x0 = ResidualsPlastic(state.εᵖ, state.κ, state.α, state.μ)
        # convert initial guess to vector
        tomandel!(cache.x_f, x0)
        # solve for variables x
        nlsolve_options = get(options, :nlsolve_params, Dict{Symbol, Any}(:method=>:newton))
        haskey(nlsolve_options, :method) || merge!(nlsolve_options, Dict{Symbol, Any}(:method=>:newton)) # set newton if the user did not supply another method
        result = NLsolve.nlsolve(cache, cache.x_f; nlsolve_options...)
        # result = NLsolve.newton(cache, cache.x_f, 0.0, 1e-8, 1000, false, false, false, LineSearches.Static(); linsolve = (x, A, b) -> copyto!(x, A\b))
        # result = NLsolve.newton_(cache, cache.x_f, 0.0, 1e-8, 1000, false, false, false, LineSearches.Static(), (x, A, b) -> copyto!(x, A\b), NLsolve.NewtonCache(cache))
        if result.f_converged
            x = frommandel(ResidualsPlastic, result.zero)
            σ = m.Eᵉ ⊡ (ε - x.εᵖ)
            dRdx = cache.DF
            inv_J_σσ = frommandel(SymmetricTensor{4,3}, inv(dRdx))
            ∂σ∂ε = inv_J_σσ ⊡ m.Eᵉ
            return σ, ∂σ∂ε, PlasticState(x.εᵖ, x.κ, x.α, x.μ)
        else
            error("Material model not converged. Could not find material state.")
        end
    end
end

function residuals(vars::ResidualsPlastic, m::Plastic, material_state::PlasticState{3}, ε)
    εᵖ = vars.εᵖ; κ = vars.κ; α = vars.α; μ = vars.μ
    σ = m.Eᵉ ⊡ (ε - εᵖ)
    ν = 3/(2*sqrt(3/2)*norm(dev(σ-α)))*dev(σ-α)
    Rσ = m.Eᵉ ⊡ (-εᵖ + material_state.εᵖ + μ * ν) # R_σ(εᵖ, α, μ)
    Rκ = κ - material_state.κ - μ*m.H_κ*(-1 +κ/m.κ_∞) # R_κ(κ, μ)
    Rα = α - material_state.α - μ*m.H_α * (-ν + 3/(2*m.α_∞)*dev(α)) # R_α(σ, α, μ)
    Rμ = sqrt(3/2)*norm(dev(σ-α)) - m.σ_y - κ
    return ResidualsPlastic(Rσ, Rκ, Rα, Rμ)
end