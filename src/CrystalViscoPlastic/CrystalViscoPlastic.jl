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
    MS::SVector{S,Tensor{2,3,Float64,9}} # m ⊗ s
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
    κ::SVector{S, T}
    α::SVector{S, T}
    μ::SVector{S, T}
end

Base.zero(::Type{CrystalViscoPlasticState{dim,T,M,S}}) where {dim,T,M,S} = CrystalViscoPlasticState{dim,T,M,S}(zero(SymmetricTensor{2,dim,T,M}), SVector{S,T}(zeros(T, S)), SVector{S,T}(zeros(T, S)), SVector{S,T}(zeros(T, S)))
initial_material_state(::CrystalViscoPlastic{S}) where S = zero(CrystalViscoPlasticState{3,Float64,6,S})

get_n_scalar_equations(::CrystalViscoPlastic{S}) where S = S

function material_response(
    m::MaterialModels.CrystalViscoPlastic{S},
    Δε::SymmetricTensor{2,3,T,6},
    state::MaterialModels.CrystalViscoPlasticState{3},
    Δt=1.0,
    args...;
    options = NamedTuple{}(),
) where {S, T}

    R(μ) = residuals(μ, m, state, Δε, Δt)
    solver_result = solve(R, state.μ; options...)
    if !isnothing(solver_result) # converged
        μ, ∂R∂μ = solver_result
        σ, κ, α = MaterialModels.get_state_vars(μ, Δε, m, state)
        # jacobian computation
        ∂R∂ε = MVector{S, typeof(Δε)}(undef)
        for i=1:S
            ∂R∂ε[i] = gradient(Δε->residuals(μ, m, state, Δε, Δt)[i], Δε)
        end
        ∂R∂μ_inv = inv(∂R∂μ)
        dσdε = m.Eᵉ
        for i=1:S
            ∂σ∂μᵢ = - m.Eᵉ ⊡ m.MS[i]*sign(σ ⊡ m.MS[i] - α[i])
            ∂μᵢ∂ε = - sum(∂R∂μ_inv[i,:] .* ∂R∂ε)
            dσdε += ∂σ∂μᵢ ⊗ ∂μᵢ∂ε
        end
        return σ, dσdε, MaterialModels.CrystalViscoPlasticState(σ, κ, α, μ)
    else
        error("Material model not converged. Could not find material state.")
    end
end

# for a reduced equation system, we need to be able to do automatic differentiation according to vars and to Δε
function residuals(
    μ::SVector{S, Tμ},
    mat::CrystalViscoPlastic{S},
    material_state::CrystalViscoPlasticState{3},
    Δε::SymmetricTensor{2,3,Tε},
    Δt) where {S,Tμ,Tε}

    T = promote_type(Tμ, Tε)
    
    σ, κ, α = get_state_vars(μ, Δε, mat, material_state)
    Rμ = MVector{S, T}(undef)
    for i=1:S
        τ = σ ⊡ mat.MS[i]
        Φ = norm(τ - α[i]) - mat.τ_y - κ[i]
        Rμ[i] = Δt*(((Φ+abs(Φ))/2)/mat.σ_c)^mat.m - mat.t_star*μ[i]
    end
        
    return SVector{S,T}(Rμ)
end

# compute other state variables based on μ
function get_state_vars(μ::SVector{S,Tμ}, Δε::SymmetricTensor{2,3,Tε}, mat::CrystalViscoPlastic{S}, material_state::CrystalViscoPlasticState{dim, T, M, S}) where {Tμ, Tε, dim, T, M, S}
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
