include("UtilityFunctions.jl")
include("Elasticity.jl")
include("IsotropicHardening.jl")
include("KinematicHardening.jl")


# Definition of material properties
struct Chaboche{T,ElasticType,IsoType,KinType} <:AbstractMaterial
    elastic::ElasticType    # Elastic definition
    σ_y0::T                 # Initial yield limit
    isotropic::IsoType      # Tuple of isotropic hardening definitions
    kinematic::KinType      # Tuple of kinematic hardening definitions
end
Chaboche(;elastic, σ_y0, isotropic, kinematic) = Chaboche(elastic, σ_y0, isotropic, kinematic)

# Definition of material state
struct ChabocheState{Nkin,T,N} <:AbstractMaterialState
    ϵₚ::SymmetricTensor{2,3,T,N}
    λ::T
    β::NTuple{Nkin, SymmetricTensor{2,3,T,N}}
end

struct ChabocheResidual{NKin,Tλ,Tσ,Tβ,N_tens}  <:AbstractResiduals
    σ::SymmetricTensor{2,3,Tσ,N_tens}
    λ::Tλ
    β::NTuple{NKin, SymmetricTensor{2,3,Tβ,N_tens}}
end

Tensors.get_base(::Type{<:ChabocheResidual{NKin}}) where{NKin} = ChabocheResidual{NKin} # needed for frommandel

function Tensors.tomandel!(v::AbstractVector{T}, r::ChabocheResidual{NKin,Tλ,Tσ,Tβ,N_tens}) where {T,NKin,Tλ,Tσ,Tβ,N_tens}
    tomandel!(v, r.σ)
    v[7] = r.λ
    for i=1:NKin
        tomandel!(v, r.β[i], offset=1+N_tens*i)
    end
    return v
end

function Tensors.frommandel(::Type{<:ChabocheResidual{NKin}}, v::AbstractVector{Tv}) where {Tv,NKin}
    σ = frommandel(SymmetricTensor{2,3}, v)
    λ = v[7]
    β = ntuple(i->frommandel(SymmetricTensor{2,3,Tv}, v, offset=1+6*i), NKin)
    return ChabocheResidual(σ,λ,β)
end

function initial_material_state(material::Chaboche{T}) where {T}
    ChabocheState(zero(SymmetricTensor{2,3,T}), 0.0, ntuple(i->zero(SymmetricTensor{2,3,T}), length(material.kinematic)))
end

# Definition of material cache
struct ChabocheCache{NL_TF, NL_TDF, NL_TX}
    R_X_oncediff::OnceDifferentiable{NL_TF, NL_TDF, NL_TX}
end

function get_cache(material::Chaboche{T,ElType,IsoType,KinType}) where {T,ElType,IsoType,KinType}
    nx = 7 + 6*length(material.kinematic)

    # Construct residual function and create OnceDifferentiable object
    state_tmp = initial_material_state(material)
    ϵ = zero(SymmetricTensor{2,3,T})
    X_tensor = initial_guess(material, state_tmp, zero(SymmetricTensor{2,3,T}))
    rf_tens(X_tensor) = residual(X_tensor, material, state_tmp, ϵ)
    rf!(R, X) = vector_residual!(rf_tens, R, X, X_tensor)
    X0 = MVector{nx}(zeros(T, nx))
    # X0 only for shape and type information here:
    R_X_oncediff = OnceDifferentiable(rf!, X0, X0; autodiff = :forward)
    return ChabocheCache(R_X_oncediff)
end

"""
    material_response(m::Chaboche, ϵ::SymmetricTensor{2,3}, state::ChabocheState, Δt; <keyword arguments>)

Return the stress tensor, stress tangent, the new `MaterialState` and a boolean specifying if local iterations converged. 
The total strain ε and previous material state `state` are given as input, Δt has no influence as the material is rate independent.

By specifying different laws in `m.elastic`, `m.isotropic`, and `m.kinematic`, different models can be obtained, 
within the general equations specified below. Note that `m.isotropic` and `m.kinematic` are of type Tuple, in which each
element contain a hardening law of type ``AbstractIsotropicHardening`` and ``AbstractKinematicHardening``, respectively. 

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

# Algorithmic tangent stiffness computations
σ(X(ϵ), ϵ): dσdϵ = ∂σ∂X dXdϵ + ∂σ∂ϵ
R(X(ϵ), ϵ): dRdX=0=∂R∂X dXdϵ + ∂R∂ϵ
dσdϵ = - ∂σ∂X [∂R∂X]^-1 ∂R∂ϵ + ∂σ∂ϵ

For this specific case, 

- ∂σ∂X is identity in upper left corner, zero the rest
- ∂R∂ϵ is the elastic stiffness in upper left corner, zero the rest
- ∂σ∂ϵ is zero (the entire stress σ is only a function of x.σ)

# Keyword arguments
- `cache`: Cache for the iterative solver, used by NLsolve.jl. It is strongly recommended to pre-allocate the cache for repeated calls to `material_response`. See [`get_cache`](@ref).
- `options::Dict{Symbol, Any}`: Solver options for the non-linear solver. Under the key `:nlsolve_params` keyword arguments for `nlsolve` can be handed over.
See [NLsolve documentation](https://github.com/JuliaNLSolvers/NLsolve.jl#common-options). By default the Newton solver will be used.
"""
function material_response(m::Chaboche, ϵ::SymmetricTensor{2,3}, old::ChabocheState{Nkin,T,N}, Δt; cache=get_cache(m), options::Dict{Symbol, Any} = Dict{Symbol, Any}()) where {T,N,Nkin}
    
    σ_trial, dσdϵ_elastic, _, _ = material_response(m.elastic, ϵ-old.ϵₚ, nothing, Δt)
    Φ_trial = vonmises(σ_trial-sum(old.β)) - (m.σ_y0 + sum(get_hardening.(m.isotropic, old.λ)))

    if Φ_trial < 0
        return σ_trial, dσdϵ_elastic, old, true
    else
        x0 = initial_guess(m, old, ϵ)
        rf!(r_vector, x_vector) = vector_residual!((x)->residual(x,m,old,ϵ), r_vector, x_vector, x0)  # Using x0 as template for residual instead of material as this is related to Tensors
        update_cache!(cache.R_X_oncediff, rf!)
        
        tomandel!(cache.R_X_oncediff.x_f, x0)
        # Should this be centrally managed? I.e. process_options or similar?
        nlsolve_options = get(options, :nlsolve_params, Dict{Symbol, Any}(:method=>:newton))
        haskey(nlsolve_options, :method) || merge!(nlsolve_options, Dict{Symbol, Any}(:method=>:newton)) # set newton if the user did not supply another method
        
        # Solve local problem:
        result = NLsolve.nlsolve(cache.R_X_oncediff, cache.R_X_oncediff.x_f; nlsolve_options...)

        if result.f_converged
            x = frommandel(ChabocheResidual{Nkin}, result.zero)
            dRdx = cache.R_X_oncediff.DF
            inv_J_σσ = frommandel(SymmetricTensor{4,3}, inv(dRdx))
            dσdϵ = inv_J_σσ ⊡ dσdϵ_elastic
            σ_red_dev = dev(x.σ) - sum(x.β)
            ϵₚ = calculate_plastic_strain(old, σ_red_dev * ((3/2)/vonmises_dev(σ_red_dev)), x.λ-old.λ)
            return x.σ, dσdϵ, ChabocheState(ϵₚ, x.λ, x.β), true
        else
            return σ_trial, dσdϵ_elastic, old, false
        end
    end
    
end

# General residual function 
function residual(x::ChabocheResidual{NKin}, m::Chaboche, old::ChabocheState, ϵ) where{NKin}
    σ_red_dev = dev(x.σ) - sum(x.β)
    σ_vm = vonmises_dev(σ_red_dev)
    Δλ = x.λ-old.λ
    ν = σ_red_dev * ((3/2)/σ_vm)                    # Gradient of von mises yield surface
    ϵₑ = calculate_elastic_strain(old, ϵ, ν, Δλ)    # Using assumption of associative plasticity
    κ = sum(get_hardening.(m.isotropic, x.λ))         

    Rσ = x.σ - calculate_sigma(m.elastic, ϵₑ)       # Using the specific elastic law
    Rλ = σ_vm - (m.σ_y0 + κ)
    Rβ = ntuple((i) -> x.β[i] - old.β[i] - Δλ * get_evolution(m.kinematic[i], ν, x.β[i]), NKin)
    
    return ChabocheResidual(Rσ, Rλ, Rβ)
end

function initial_guess(m::Chaboche, old::ChabocheState{Nkin}, ϵ) where {Nkin}
    σ_trial = calculate_sigma(m.elastic, ϵ-old.ϵₚ)
    λ = old.λ
    β = ntuple(i->old.β[i], Nkin)
    return ChabocheResidual(σ_trial,λ,β)
end

function calculate_sigma(m::LinearIsotropicElasticity, ϵₑ)
    return 3*m.K*vol(ϵₑ) + 2 * m.G * dev(ϵₑ)
end

function calculate_elastic_strain(old::ChabocheState, ϵ, ν, Δλ)
    return ϵ - calculate_plastic_strain(old, ν, Δλ)
end

function calculate_plastic_strain(old::ChabocheState, ν, Δλ)
    return old.ϵₚ + Δλ*ν
end
