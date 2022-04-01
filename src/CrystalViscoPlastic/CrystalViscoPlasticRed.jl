"""
CrystalViscoPlasticRed(E, ν, τ_y, H_iso, H_kin, q, α_∞, t_star, σ_c, m, slipsystems)

Crystal plasticity with 
# Arguments
- `E::Float64`: Young's modulus
- `ν::Float64`: Poisson's ratio
- `τ_y`: critical resolved shear stress
- `H_iso`: hardening modulus for isotropic hardening
- `H_kin`: hardening modulus for kinematic hardening
- `q`: cross-hardening parameter for isotropic hardening
- `α_∞`: saturation stress for kinematic hardening
- `t_star`: visco-plastic characteristic time
- `σ_c`: visco-plastic characteristic stress
- `m`: visco-plastic exponent
- `slipsystems`: slipsystems
"""
struct CrystalViscoPlasticRed{S} <: AbstractMaterial
    E::Float64
    ν::Float64
    τ_y::Float64
    H_iso::Float64
    H_kin::Float64
    q::Float64
    α_∞::Float64
    t_star::Float64
    σ_c::Float64
    m::Float64

    # precomputed elastic tangent tensor and hardening matrix
    Eᵉ::SymmetricTensor{4,3,Float64,36}
    H::Matrix{Float64}
    MS::SVector{S,Tensor{2,3,Float64,9}} # m ⊗ s
    # store information about slip sytems
    nslipsystems::Int
    slipsystems::Vector{NTuple{2, Vec{3}}}

    function CrystalViscoPlasticRed(E, ν, τ_y, H_iso, H_kin, q, α_∞, t_star, σ_c, m, slipsystems)
        Eᵉ = elastic_tangent_3D(E, ν)
        nslipsystems = length(slipsystems)
        H = H_iso*(q*ones(nslipsystems, nslipsystems) + (1-q)*diagm(ones(nslipsystems)))
        MS = SVector{nslipsystems,Tensor{2,3,Float64,9}}([slipsystems[i][1] ⊗ slipsystems[i][2] for i=1:nslipsystems])
        return new{nslipsystems}(E, ν, τ_y, H_iso, H_kin, q, α_∞, t_star, σ_c, m, Eᵉ, H, MS, nslipsystems, slipsystems)
    end
end

# keyword argument constructor
CrystalViscoPlasticRed(; E, ν, τ_y, H_iso, H_kin, q, α_∞, t_star, σ_c, m, slipsystems) = CrystalViscoPlasticRed(E, ν, τ_y, H_iso, H_kin, q, α_∞, t_star, σ_c, m, slipsystems)

get_n_slipsystems(::CrystalViscoPlasticRed{S}) where S = S

struct CrystalViscoPlasticRedState{dim,T,M,S} <: AbstractMaterialState
    σ::SymmetricTensor{2,dim,T,M}
    κ::SVector{S, T}
    α::SVector{S, T}
    μ::SVector{S, T}
end

Base.zero(::Type{CrystalViscoPlasticRedState{dim,T,M,S}}) where {dim,T,M,S} = CrystalViscoPlasticRedState{dim,T,M,S}(zero(SymmetricTensor{2,dim,T,M}), SVector{S,T}(zeros(T, S)), SVector{S,T}(zeros(T, S)), SVector{S,T}(zeros(T, S)))
initial_material_state(::CrystalViscoPlasticRed{S}) where S = zero(CrystalViscoPlasticRedState{3,Float64,6,S})

function get_cache(m::CrystalViscoPlasticRed)
    state = initial_material_state(m)
    # it doesn't actually matter which state and strain step we use here,
    # f is overwritten in the constitutive driver before being used.
    f(r_vector, x_vector) = vector_residual!(((x)->MaterialModels.residuals(x, m, state, zero(SymmetricTensor{2,3}), 1.0)), r_vector, x_vector, m)
    v_cache = Vector{Float64}(undef, get_n_scalar_equations(m))
    cache = NLsolve.OnceDifferentiable(f, v_cache, v_cache; autodiff = :forward)
    return cache
end

# constitutive driver operates in 3D, so these can always be 3D
# TODO: Should Residuals have a Type Parameter N for the number of scalar equations?
struct ResidualsCrystalViscoPlasticRed{S, T}
    μ::SVector{S, T}
end

get_n_scalar_equations(::CrystalViscoPlasticRed{S}) where S = S
Tensors.get_base(::Type{CrystalViscoPlasticRed{S}}) where S = ResidualsCrystalViscoPlasticRed{S} # needed for frommandel

function Tensors.tomandel!(v::Vector{T}, r::ResidualsCrystalViscoPlasticRed{S, T}) where {S, T}
    v[1:S] = view(r.μ, :)
    return v
end

function Tensors.frommandel(::Type{ResidualsCrystalViscoPlasticRed{S}}, v::AbstractVector{T}) where {S, T}
    μ = SVector{S,T}(view(v, 1:S))
    return ResidualsCrystalViscoPlasticRed(μ)
end

function material_response(m::CrystalViscoPlasticRed{S}, Δε::SymmetricTensor{2,3,T,6}, state::CrystalViscoPlasticRedState{3},
    Δt=1.0; cache=get_cache(m), options::Dict{Symbol, Any} = Dict{Symbol, Any}()) where {S, T}

    # set the current residual function that depends only on the variables
    f(r_vector, x_vector) = vector_residual!((x->residuals(x,m,state,Δε,Δt)), r_vector, x_vector, m)
    update_cache!(cache, f)
    # initial guess
    σ_trial = state.σ + m.Eᵉ ⊡ Δε
    x0 = ResidualsCrystalViscoPlasticRed(state.μ)
    # convert initial guess to vector
    tomandel!(cache.x_f, x0)
    # solve for variables x
    nlsolve_options = get(options, :nlsolve_params, Dict{Symbol, Any}(:method=>:newton))
    haskey(nlsolve_options, :method) || merge!(nlsolve_options, Dict{Symbol, Any}(:method=>:newton)) # set newton if the user did not supply another method
    result = NLsolve.nlsolve(cache, cache.x_f; nlsolve_options...)
    
    if result.f_converged
        x_mandel = result.zero::Vector{T}
        x = frommandel(ResidualsCrystalViscoPlasticRed{S}, x_mandel)
        # compute other state variables
        σ, κ, α = get_state_vars(x.μ, Δε, m, state)
        ############################################
        # tangent computation
        ∂R∂x = cache.DF
        f2(r_vector, x_vector) = vector_residual!((Δε->residuals(x,m,state,Δε,Δt)), r_vector, x_vector, Δε)
        # lucky coincidence that we can reuse the buffers here
        tomandel!(view(cache.x_f, 1:6), Δε)
        ∂R∂ε = ForwardDiff.jacobian(f2, cache.F, view(cache.x_f, 1:6)) # ∂R∂ε = ∂R∂Δε
        ∂X∂ε = -inv(∂R∂x)*∂R∂ε 
        # in this case X is only μ
        ∂σ∂ε = m.Eᵉ
        for i=1:S
            ∂μᵢ∂ε = frommandel(SymmetricTensor{2,3}, ∂X∂ε[i,:])
            # we've already done almost all the computations we need here - can they be reused?
            ∂σ∂μᵢ = -m.Eᵉ ⊡ m.MS[i]*sign(σ ⊡ m.MS[i] - α[i])
            ∂σ∂ε += ∂σ∂μᵢ ⊗ ∂μᵢ∂ε
        end
        return σ, ∂σ∂ε, CrystalViscoPlasticRedState(σ, κ, α, x.μ)
    else
        error("Material model not converged. Could not find material state.")
    end
end

# for a reduced equation system, we need to be able to do automatic differentiation according to vars and to Δε
function residuals(vars::ResidualsCrystalViscoPlasticRed{S,Tv}, mat::CrystalViscoPlasticRed{S}, material_state::CrystalViscoPlasticRedState{3}, Δε::SymmetricTensor{2,3,Tε}, Δt) where {S,Tv,Tε}
    T = promote_type(Tv, Tε)
    μ=vars.μ
    
    σ, κ, α = get_state_vars(μ, Δε, mat, material_state)
    Rμ = MVector{S, T}(undef)
    for i=1:S
        τ = σ ⊡ mat.MS[i]
        Φ = norm(τ - α[i]) - mat.τ_y - κ[i]
        Rμ[i] = Δt*(((Φ+abs(Φ))/2)/mat.σ_c)^mat.m - mat.t_star*μ[i]
    end
        
    return  ResidualsCrystalViscoPlasticRed(SVector{S,T}(Rμ))
end

# compute other state variables based on μ
function get_state_vars(μ::SVector{S,Tμ}, Δε::SymmetricTensor{2,3,Tε}, mat::CrystalViscoPlasticRed{S}, material_state::CrystalViscoPlasticRedState{dim, T, M, S}) where {Tμ, Tε, dim, T, M, S}
    σ_trial = material_state.σ + mat.Eᵉ ⊡ Δε
    τ_trial = map(ms -> σ_trial ⊡ ms, mat.MS)
    sign_τ_red_trial = sign.(τ_trial - material_state.α)
    temp_sum_σ = zero(SymmetricTensor{2,3})
    κ = MVector{S, Tμ}(undef)
    α = MVector{S, promote_type(Tμ, Tε)}(undef)
    for i=1:S
        temp_sum_σ += μ[i]*sign_τ_red_trial[i]*mat.MS[i]
        κ[i] = material_state.κ[i] + dot(view(mat.H, :, i), μ)
        α[i] = (material_state.α[i] + mat.H_kin*μ[i]*sign_τ_red_trial[i])/(1+mat.H_kin*μ[i]/mat.α_∞)
    end
    σ = σ_trial - mat.Eᵉ ⊡ temp_sum_σ
    return σ, SVector{S,Tμ}(κ), SVector{S,promote_type(Tμ,Tε)}(α)
end

