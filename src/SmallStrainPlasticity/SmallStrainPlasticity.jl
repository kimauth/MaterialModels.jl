include("UtilityFunctions.jl")
include("Elasticity.jl")
include("IsotropicHardening.jl")
include("KinematicHardening.jl")


# Definition of material properties
struct Chaboche{T,ElasticType,IsoType,KinType}
    elastic::ElasticType    # Elastic definition
    σ_y0::T                 # Initial yield limit
    isotropic::IsoType      # Tuple of isotropic hardening definitions
    kinematic::KinType      # Tuple of kinematic hardening definitions
end
Chaboche(;elastic, σ_y0, isotropic, kinematic) = Chaboche(elastic, σ_y0, isotropic, kinematic)

# Definition of material state
struct ChabocheState{Nkin,T,N}
    ϵₚ::SymmetricTensor{2,3,T,N}
    λ::T
    β::NTuple{Nkin, SymmetricTensor{2,3,T,N}}
end

struct ChabocheResidual{NKin_R,Tλ,Tσ,Tβ,N_tens}
    λ::Tλ
    σ_red_dev::SymmetricTensor{2,3,Tσ,N_tens}
    β1::NTuple{NKin_R, SymmetricTensor{2,3,Tβ,N_tens}}
end
# Specialize for only one backstress
function ChabocheResidual(λ::Tλ,σ_red_dev::SymmetricTensor{2,3,Tσ,N_tens}) where {Tλ,Tσ,N_tens}
    ChabocheResidual{0,Tλ,Tσ,Float64,N_tens}(λ,σ_red_dev,())
end

Tensors.get_base(::Type{<:ChabocheResidual{NKin_R}}) where{NKin_R} = ChabocheResidual{NKin_R} # needed for frommandel

function Tensors.tomandel!(v::AbstractVector{T}, r::ChabocheResidual{NKin_R,Tλ,Tσ,Tβ,N_tens}) where {T,NKin_R,Tλ,Tσ,Tβ,N_tens}
    v[1] = r.λ
    tomandel!(v, r.σ_red_dev, offset=1)
    for i=1:NKin_R
        tomandel!(v, r.β1[i], offset=1+N_tens*i)
    end
    return v
end

function Tensors.frommandel(::Type{<:ChabocheResidual{NKin_R}}, v::AbstractVector{Tv}) where {Tv,NKin_R}
    λ = v[1]
    σ_red_dev = frommandel(SymmetricTensor{2,3}, v, offset=1)
    if NKin_R > 0
        β1 = ntuple(i->frommandel(SymmetricTensor{2,3,Tv}, v, offset=1+6*i), NKin_R)
        return ChabocheResidual(λ,σ_red_dev,β1)
    else
        return ChabocheResidual(λ,σ_red_dev)
    end
end

function initial_material_state(material::Chaboche{T}) where {T}
    ChabocheState(zero(SymmetricTensor{2,3,T}), 0.0, ntuple(i->zero(SymmetricTensor{2,3,T}), Val{length(material.kinematic)}()))
end

# Definition of material cache
##=
struct ChabocheCache{T, nx, nx6, NL_TF, NL_TDF, NL_TX, NL_NC}
    # General purpose
    vnx6::MMatrix{nx,6,T,nx6}
    v6xn::MMatrix{6,nx,T,nx6}
    v6x6::MMatrix{6,6,T,36}
    v6::MVector{6,T}
    # For solving R(X)=0
    #X0::MVector{nx,T}
    R_X_oncediff::OnceDifferentiable{NL_TF, NL_TDF, NL_TX}
    R_X_newton::NLsolve.NewtonCache{NL_NC}
end

function get_cache(material::Chaboche{T,ElType,IsoType,KinType}) where {T,ElType,IsoType,KinType}
    nx = 1 + 6*length(material.kinematic)

    # Construct residual function and create OnceDifferentiable object
    state_tmp = initial_material_state(material)
    σ_trial_dev = zero(SymmetricTensor{2,3,T})
    X_tensor = initial_guess(material, state_tmp, zero(SymmetricTensor{2,3,T}))
    rf_tens(X_tensor) = residual(X_tensor, material, state_tmp, σ_trial_dev)
    rf!(R, X) = vector_residual!(rf_tens, R, X, X_tensor)
    X0 = MVector{nx}(zeros(T, nx))
    # X0 only for shape and type information here:
    R_X_oncediff = OnceDifferentiable(rf!, X0, X0; autodiff = :forward)
    R_X_newton = NLsolve.NewtonCache(R_X_oncediff)
    return ChabocheCache(MMatrix{nx,6}(zeros(nx,6)), MMatrix{6,nx}(zeros(6,nx)), MMatrix{6,6}(zeros(6,6)), MVector{6}(zeros(6)),
                         R_X_oncediff, R_X_newton)
end

"""
    material_response(m::Chaboche, ϵ::SymmetricTensor{2,3}, state::ChabocheState, Δt; <keyword arguments>)

Return the stress tensor, stress tangent, the new `MaterialState` and a boolean specifying if local iterations converged. 
The total strain ε and previous material state `state` are given as input, Δt has no influence as the material is rate independent.

By specifying different laws in `m.elastic`, `m.isotropic`, and `m.kinematic`, different models can be obtained, 
within the general equations specified below. Note that `m.isotropic` and `m.kinematic` are of type Tuple, in which each
element contain a hardening law of type ``AbstractiIsoHard`` and ``AbstractKinHard``, respectively. 

The stress is calculated from the elastic strains, ``\\boldsymbol{\\epsilon}_\\mathrm{e}``, obtained via the 
additive decomposition, ``\\boldsymbol{\\epsilon} = \\boldsymbol{\\epsilon}_\\mathrm{e} + \\boldsymbol{\\epsilon}_\\mathrm{p}``. 
The elastic law, i.e. `` is specified by `m.elastic`

Von Mises yield function:
```math
\\Phi = \\sqrt{\\frac{3}{2}} \\left| \\text{dev} \\left( \\boldsymbol{\\sigma} - \\boldsymbol{\\beta} \\right) \\right| - \\sigma_y - \\kappa
```
where ``\\boldsymbol{\\beta} = \\sum_{i=1}^{N_\\mathrm{kin} \\boldsymbol{\\beta}_i`` is the total back-stress. 
The evolution of ``\\boldsymbol{\\beta}_i`` is given by the kinematic hardening, specified below and using `m.kinematic`

Associative plastic flow is used to obtain the plastic strains,
```math
\\dot{\\epsilon}_{\\mathrm{p}} = \\dot{\\lambda} \\frac{\\partial \\Phi}{\\boldsymbol{\\sigma}}
= \\dot{\\lambda} \\boldsymbol{\\nu}
```

The plastic multiplier, ``\\lambda``, is obtained via the KKT-conditions,
```math
\\dot{\\lambda} \\geq 0, \\quad \\Phi \\leq 0, \\quad \\dot{\\lambda}\\Phi = 0
```

The isotropic hardening is formulated as
```math
\\kappa = \\sum_{i=1}^{N_{\\mathrm{iso}}} g_{\\mathrm{iso},i}(\\lambda)
```
where ``g_{\\mathrm{iso},i}(\\lambda)`` is specified by `m.isotropic[i]`

Kinematic hardening is formulated as
```math
\\boldsymbol{\\beta}_i = \\dot{\\lambda} g_{\\mathrm{kin},i}(\\nu, \\boldsymbol{\\beta}_i)
```
where ``g_{\\mathrm{kin},i}(\\boldsymbol{\\nu}, \\boldsymbol{\\beta}_i)`` is specified by `m.kinematic[i]`
and ``i\\in[1,N_\\mathrm{kin}]``. 

```
# Keyword arguments
- `cache`: Cache for the iterative solver, used by NLsolve.jl. It is strongly recommended to pre-allocate the cache for repeated calls to `material_response`. See [`get_cache`](@ref).
- `options::Dict{Symbol, Any}`: Solver options for the non-linear solver. Under the key `:nlsolve_params` keyword arguments for `nlsolve` can be handed over.
See [NLsolve documentation](https://github.com/JuliaNLSolvers/NLsolve.jl#common-options). By default the Newton solver will be used.
"""
function material_response(material::Chaboche, ϵ::SymmetricTensor{2,3}, state_old::ChabocheState{Nkin,T,N}, Δt; cache=get_cache(material), options::Dict{Symbol, Any} = Dict{Symbol, Any}()) where {T,N,Nkin}
    
    σ_trial, dσdϵ_elastic, _, _ = material_response(material.elastic, ϵ-state_old.ϵₚ, nothing, Δt)

    Φ_trial = yieldCriterion(material, σ_trial-sum(state_old.β), state_old.λ)
    if Φ_trial < 0
        return σ_trial, dσdϵ_elastic, state_old, true
    else
        num_blas_threads = LinearAlgebra.BLAS.get_num_threads() # (~ 2ns)
        LinearAlgebra.BLAS.set_num_threads(1)                   # Big performance benefit, takes ~8ns
        converged = solve_local_problem!(cache, material, state_old, ϵ, options)
        if converged
            σ, dσdϵ, state = get_plastic_output(cache, material, state_old, σ_trial, dσdϵ_elastic)
        else
            σ, dσdϵ, state = (σ_trial, dσdϵ_elastic, state_old)
            println("Did not converge!")
        end
        LinearAlgebra.BLAS.set_num_threads(num_blas_threads)
        return σ, dσdϵ, state, converged
    end
    
end

include("Residual.jl")

function get_plastic_output(cache::ChabocheCache, material::Chaboche, state_old::ChabocheState{NKin,Ts,N}, ϵ::SymmetricTensor{2,3}, dσdϵ_elastic::SymmetricTensor{4,3}) where {NKin,Ts,N}
    # σ = σ(X(ϵ), ϵ) yields
    # dσ/dϵ = ∂σ/∂ϵ + ∂σ/∂X : dX/dϵ              [1]
    # R = R(X(ϵ), ϵ) yields
    # dR/dϵ = 0 = ∂R/∂ϵ + ∂R/∂X : dXdϵ           [2]
    # Solve [2] for dX/dϵ
    # dX/dϵ = - [∂R/∂X]^-1 : ∂R∂ϵ                [3]
    # Insert [3] in [1]
    # dσ/dϵ = ∂σ/∂ϵ - ∂σ/∂X : [∂R/∂X]^-1 : ∂R/∂ϵ [4]
    
    X_vec = cache.R_X_oncediff.x_f
    dRdX = cache.R_X_oncediff.DF
    X_tensor = frommandel(ChabocheResidual{NKin-1,Ts,Ts,Ts,N}, X_vec)

    # ∂σ∂X
    # - Stress function
    σ_X(X_arg) = get_sigma(material, state_old, X_arg, ϵ)
    # - Stress (vector) function:
    σ_X_vec!(σv_arg, Xv_arg) = vector_residual!(σ_X, σv_arg, Xv_arg, X_tensor)
    σ_vec = cache.v6
    # - Preallocate (should have been done beforehand, problem with Dual Tag values?)
    cfg = ForwardDiff.JacobianConfig(σ_X_vec!, σ_vec, X_vec, ForwardDiff.Chunk{length(X_vec)}())
    # - Create DiffResult (this should be non-allocating)
    ∂σ∂X = cache.v6xn
    diff_result = DiffResults.MutableDiffResult(σ_vec, (∂σ∂X,))
    # - Calculate σ and ∂σ∂X
    ForwardDiff.jacobian!(diff_result, σ_X_vec!, σ_vec, X_vec, cfg)
    σ = frommandel(SymmetricTensor{2,3}, diff_result.value)

    # ∂R/∂ϵ
    # - Specialized residual (tensor) function:
    R_ϵ(ϵ_arg) = residual(X_tensor, material, state_old, ϵ_arg)
    # - Specialized residual (vector) function:
    R_ϵ_vec!(Rv_arg, ϵv_arg) = vector_residual!(R_ϵ, Rv_arg, ϵv_arg, ϵ)
    ϵ_vec = cache.v6    # Use cache value (give name that makes more sense)
    tomandel!(ϵ_vec, ϵ)
    # - Preallocate (should have been done beforehand, problem with Dual Tag values?)
    R_vec = X_vec   # Use as cache (ok as X_vec is not used anymore)
    cfg = ForwardDiff.JacobianConfig(R_ϵ_vec!, R_vec, ϵ_vec, ForwardDiff.Chunk{length(6)}())
    # - Calculate ∂R∂ϵ
    ∂R∂ϵ = cache.vnx6
    ForwardDiff.jacobian!(∂R∂ϵ, R_ϵ_vec!, R_vec, ϵ_vec, cfg)

    # Calculate full tangent stiffness 
    dσdϵ = dσdϵ_elastic - frommandel(SymmetricTensor{4,3}, ∂σ∂X*(dRdX\∂R∂ϵ))
    
    λ = X_tensor.λ
    Δλ = λ-state_old.λ
    σ_red_dev = X_tensor.σ_red_dev
    ν = (3.0/2.0) * σ_red_dev / vonMisesDev(σ_red_dev)
    
    if NKin > 1
        β = ntuple(i-> i==1 ? dev(σ) - σ_red_dev - sum(X_tensor.β1) : X_tensor.β1[i-1], NKin)
    else
        β = (dev(σ) - σ_red_dev,)
    end
    
    ϵₚ = state_old.ϵₚ + Δλ * ν
    state = ChabocheState(ϵₚ, λ, β)
    
    return σ, dσdϵ, state
end

function initial_guess(material::Chaboche, state_old::ChabocheState{Nkin, T, N}, ϵ) where {T, Nkin, N}
    if Nkin<1
        error("Nkin < 1 is not supported")
    end
    # Becomes trial by setting Δλ=0
    σ_trial_dev = calc_sigma_dev(material.elastic, state_old, ϵ, ϵ, 0.0)
    λ = state_old.λ
    σ_red_trial = σ_trial_dev - sum(state_old.β)
    
    if Nkin > 1
        return ChabocheResidual(λ,σ_red_trial,ntuple(i->state_old.β[i], Val{Nkin-1}()))
    else
        return ChabocheResidual(λ,σ_red_trial)
    end
end


function solve_local_problem!(cache::ChabocheCache, material::Chaboche, state_old::ChabocheState{Nkin,Ts,N}, ϵ::SymmetricTensor{2,3},  options::Dict{Symbol, Any}) where {Ts,Nkin,N}
    
    X_tensor = initial_guess(material, state_old, ϵ)
    rf_tens(X_tensor_arg) = residual(X_tensor_arg, material, state_old, ϵ)
    rf!(R, X) = vector_residual!(rf_tens, R, X, X_tensor)
    update_cache!(cache.R_X_oncediff, rf!)
    
    tomandel!(cache.R_X_oncediff.x_f, X_tensor)
    # Should this be centrally managed? I.e. process_options or similar?
    nlsolve_options = get(options, :nlsolve_params, Dict{Symbol, Any}(:method=>:newton))
    haskey(nlsolve_options, :method) || merge!(nlsolve_options, Dict{Symbol, Any}(:method=>:newton)) # set newton if the user did not supply another method
    nlsolve_options[:method] == :newton || merge!(nlsolve_options, Dict(:cache=>cache.R_X_newton))
    # Need to call newton directly to allow newton caching...
    #result = NLsolve.newton(; ftol=1.e-6, cache=cache.R_X_newton)
    my_newton(df, x0; xtol=0.0, ftol=1.e-8, iterations=100, store_trace=false, show_trace=false, extended_trace=false, linesearch=NLsolve.LineSearches.Static(),cache=NewtonCache(df)) = NLsolve.newton(df, x0, xtol, ftol, iterations, store_trace, show_trace, extended_trace, linesearch,cache)
    result = my_newton(cache.R_X_oncediff, cache.R_X_oncediff.x_f, cache=cache.R_X_newton)
    # Is this necessary?
    cache.R_X_oncediff.x_f = result.zero
    
    return result.f_converged
end

function get_sigma(material::Chaboche, state_old::ChabocheState, X::ChabocheResidual, ϵ::SymmetricTensor{2,3})
    Δλ = X.λ - state_old.λ
    σ_red_dev = X.σ_red_dev
    ν = (3.0/2.0) * σ_red_dev / vonMises(σ_red_dev)
    σ = calc_sigma(material.elastic, state_old, ϵ, ν, Δλ)
    return σ
end

function calc_sigma(material::LinearIsotropicElasticity, state_old::ChabocheState, ϵ, ν, Δλ)
    return 3 * material.K*vol(ϵ) + calc_sigma_dev(material, state_old, ϵ, Δλ, ν)
end

function calc_sigma_dev(material::LinearIsotropicElasticity, state_old::ChabocheState, ϵ, ν, Δλ)
    return 2 * material.G * (dev(ϵ - state_old.ϵₚ) - Δλ*ν)
end

function yieldCriterion(material::Chaboche{Tp,ElType,IsoType,KinType}, σ_vm_red::Number, λ) where {Tp,ElType,IsoType<:NTuple{Niso,Any},KinType} where {Niso}
    κ = sum(ntuple(i->IsotropicHardening(material.isotropic[i], λ), Val{Niso}()))
    return σ_vm_red - (κ + material.σ_y0)
end

function yieldCriterion(material::Chaboche, σ_red_dev::AbstractTensor, λ)
    return yieldCriterion(material, vonMises(σ_red_dev), λ)
end
