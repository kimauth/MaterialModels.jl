"""
CrystalViscoPlastic(E, ν, τ_y, H_iso, H_kin, q, α_∞, t_star, σ_c, m, slipsystems)

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
struct CrystalViscoPlastic{S} <: AbstractMaterial
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
    MS::SVector{S,Tensor{2,3,Float64,9}}
    # store information about slip sytems
    nslipsystems::Int
    slipsystems::Vector{NTuple{2, Vec{3}}}

    function CrystalViscoPlastic(E, ν, τ_y, H_iso, H_kin, q, α_∞, t_star, σ_c, m, slipsystems)
        Eᵉ = elastic_tangent_3D(E, ν)
        nslipsystems = length(slipsystems)
        H = H_iso*(q*ones(nslipsystems, nslipsystems) + (1-q)*diagm(ones(nslipsystems)))
        MS = SVector{nslipsystems,Tensor{2,3,Float64,9}}([slipsystems[i][1] ⊗ slipsystems[i][2] for i=1:nslipsystems])
        return new{nslipsystems}(E, ν, τ_y, H_iso, H_kin, q, α_∞, t_star, σ_c, m, Eᵉ, H, MS, nslipsystems, slipsystems)
    end
end

# keyword argument constructor
CrystalViscoPlastic(; E, ν, τ_y, H_iso, H_kin, q, α_∞, t_star, σ_c, m, slipsystems) = CrystalViscoPlastic(E, ν, τ_y, H_iso, H_kin, q, α_∞, t_star, σ_c, m, slipsystems)

get_n_slipsystems(::CrystalViscoPlastic{S}) where S = S

struct CrystalViscoPlasticState{dim,T,M,S} <: AbstractMaterialState
    σ::SymmetricTensor{2,dim,T,M}
    κ::SVector{S, Float64}
    α::SVector{S, Float64}
    μ::SVector{S, Float64}
end

Base.zero(::Type{CrystalViscoPlasticState{dim,T,M,S}}) where {dim,T,M,S} = CrystalViscoPlasticState{dim,T,M,S}(zero(SymmetricTensor{2,dim,T,M}), SVector{S,T}(zeros(T, S)), SVector{S,T}(zeros(T, S)), SVector{S,T}(zeros(T, S)))
initial_material_state(::CrystalViscoPlastic{S}) where S = zero(CrystalViscoPlasticState{3,Float64,6,S})

function get_cache(m::CrystalViscoPlastic)
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
struct ResidualsCrystalViscoPlastic{S, T}
    σ::SymmetricTensor{2,3,T,6}
    κ::SVector{S, T}
    α::SVector{S, T}
    μ::SVector{S, T}
    # function ResidualsCrystalViscoPlastic(σ::SymmetricTensor{2,3,T,6}, κ::MVector{S, T}, α::MVector{S, T}, μ::MVector{S, T}) where {S, T}
    #     # S = length(κ)
    #     # length(α) == S || error("$S slipsystems for κ, but $(length(α)) slipsystems for α handed over to Residuals{CrystalViscoPlastic}.")
    #     # length(μ) == S || error("$S slipsystems for κ, but $(length(μ)) slipsystems for μ handed over to Residuals{CrystalViscoPlastic}.")
    #     new{S,T}(σ, κ, α, μ)
    # end
end

get_n_scalar_equations(::CrystalViscoPlastic{S}) where S = 6 + 3S
# get_residual_type(::CrystalViscoPlastic{S}) where S = ResidualsCrystalViscoPlastic{S}
Tensors.get_base(::Type{CrystalViscoPlastic{S}}) where S = ResidualsCrystalViscoPlastic{S} # needed for frommandel

function Tensors.tomandel!(v::Vector{T}, r::ResidualsCrystalViscoPlastic{S, T}) where {S, T}
    M=6
    # TODO check vector length
    tomandel!(view(v, 1:M), r.σ)
    v[M+1:M+S] = view(r.κ, :)
    v[M+S+1:M+2S] = view(r.α, :)
    v[M+2S+1:M+3S] = view(r.μ, :)
    return v
end

function Tensors.frommandel(::Type{ResidualsCrystalViscoPlastic{S}}, v::Vector{T}) where {S, T}
    M = 6
    σ = frommandel(SymmetricTensor{2,3}, view(v, 1:M))
    κ = SVector{S,T}(view(v, M+1:M+S))
    α = SVector{S,T}(view(v, M+S+1:M+2S))
    μ = SVector{S,T}(view(v, M+2S+1:M+3S))
    return ResidualsCrystalViscoPlastic(σ, κ, α, μ)
end

function material_response(m::CrystalViscoPlastic{S}, Δε::SymmetricTensor{2,3,T,6}, state::CrystalViscoPlasticState{3},
    Δt=1.0; cache=get_cache(m), options::Dict{Symbol, Any} = Dict{Symbol, Any}()) where {S, T}

    # set the current residual function that depends only on the variables
    f(r_vector, x_vector) = vector_residual!((x->residuals(x,m,state,Δε,Δt)), r_vector, x_vector, m)
    update_cache!(cache, f)
    # initial guess
    σ_trial = state.σ + m.Eᵉ ⊡ Δε
    x0 = ResidualsCrystalViscoPlastic(σ_trial, state.κ, state.α, state.μ)
    # convert initial guess to vector
    tomandel!(cache.x_f, x0)
    # solve for variables x
    nlsolve_options = get(options, :nlsolve_params, Dict{Symbol, Any}(:method=>:newton))
    haskey(nlsolve_options, :method) || merge!(nlsolve_options, Dict{Symbol, Any}(:method=>:newton)) # set newton if the user did not supply another method
    result = NLsolve.nlsolve(cache, cache.x_f; nlsolve_options...)
    
    if result.f_converged
        x_mandel = result.zero::Vector{T}
        x = frommandel(ResidualsCrystalViscoPlastic{S}, x_mandel)
        dRdx = cache.DF
        inv_J_σσ = frommandel(SymmetricTensor{4,3}, inv(dRdx))
        ∂σ∂ε = inv_J_σσ ⊡ m.Eᵉ
        return x.σ, ∂σ∂ε, CrystalViscoPlasticState{3,T,6,S}(x.σ, SVector{S,T}(x.κ), SVector{S,T}(x.α), SVector{S,T}(x.μ))
    else
        error("Material model not converged. Could not find material state.")
    end
end

function residuals(vars::ResidualsCrystalViscoPlastic{S,T}, mat::CrystalViscoPlastic{S}, material_state::CrystalViscoPlasticState{3}, Δε, Δt) where {S,T}
    σ=vars.σ; κ=vars.κ; α=vars.α; μ=vars.μ

    temp_sum_σ = zero(Tensor{2,3})
    Rκ = MVector{S, T}(undef)
    Rα = MVector{S, T}(undef)
    Rμ = MVector{S, T}(undef)
    for i=1:S
        # m = mat.slipsystems[i][1]
        # s = mat.slipsystems[i][2]
        # ms_tensor = m ⊗ s
        ms_tensor = mat.MS[i]
        τ = σ ⊡ ms_tensor
        σ_red = τ - α[i]
        stress_sign = sign(σ_red)
        Φ = norm(σ_red) - (mat.τ_y + κ[i])

        # R_σ
        temp_sum_σ += μ[i] * ms_tensor * stress_sign

        # R_κ
        # temp_sum_κ = 0.0
        # for j=1:S
        #     # H is symmetric, thus we can take columns instead of rows
        #     temp_sum_κ += mat.H[j,i]*μ[i]
        # end
        Rκ[i] = κ[i] - material_state.κ[i] - dot(view(mat.H, :,i), μ)

        # R_α
        Rα[i] = α[i] - material_state.α[i] - μ[i]*mat.H_kin*(stress_sign-α[i]/mat.α_∞)

        # R_Φ
        Rμ[i] = Δt*(((Φ+abs(Φ))/2)/mat.σ_c)^mat.m - mat.t_star*μ[i]
    end

    Rσ = σ - material_state.σ - mat.Eᵉ ⊡ Δε + mat.Eᵉ ⊡ temp_sum_σ
    return  ResidualsCrystalViscoPlastic(Rσ, SVector{S,T}(Rκ), SVector{S,T}(Rα), SVector{S,T}(Rμ))
end

# # use StaticVectors everywhere
# function residuals_var(vars::ResidualsCrystalViscoPlastic{S,T}, mat::CrystalViscoPlastic{S}, material_state::CrystalViscoPlasticState{3}, Δε, Δt) where {S,T}
#     σ=vars.σ; κ=vars.κ; α=vars.α; μ=vars.μ

#     temp_sum_σ = zero(Tensor{2,3})
#     for i=1:S
#         m = mat.slipsystems[i][1]
#         s = mat.slipsystems[i][2]
#         ms_tensor = m ⊗ s
#         τ = σ ⊡ ms_tensor
        
#         stress_sign = sign(τ - α[i])
#         Φ = norm(τ - α[i]) - (mat.τ_y + κ[i])

#         # R_σ
#         temp_sum_σ += μ[i] * ms_tensor * stress_sign

#         # R_κ
#         temp_sum_κ = 0.0
#         for j=1:S
#             # H is symmetric, thus we can take columns instead of rows
#             temp_sum_κ += mat.H[j,i]*μ[i]
#         end
#         Rκ[i] = κ[i] - material_state.κ[i] - temp_sum_κ

#         # R_α
#         Rα[i] = α[i] - material_state.α[i] - μ[i]*mat.H_kin*(stress_sign-α[i]/mat.α_∞)

#         # R_Φ
#         Rμ[i] = Δt*(((Φ+abs(Φ))/2)/mat.σ_c)^mat.m - mat.t_star*μ[i]
#     end

#     Rσ = σ - material_state.σ - mat.Eᵉ ⊡ Δε + mat.Eᵉ ⊡ temp_sum_σ
#     return  ResidualsCrystalViscoPlastic(Rσ, Rκ, Rα, Rμ)
# end