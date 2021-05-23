"""
CrystalViscoPlastic(E, ν, τ_y, H_iso, H_kin, q, α_∞, t_star, σ_c, m, slipsystems)

Crystal plasticity with 
# Arguments
- `E::Float64`: Young's modulus
- `ν::Float64`: Poisson's ratio
- `τ_y`: yield limit
- `H_iso`: hardening modulus for isotropic hardening
- `H_kin`: hardening modulus for kinematic hardening
- `q`: cross-hardening parameter for isotropic hardening
- `α_∞`: saturation stress for kinematic hardening
- `t_star`: visco-plastic characteristic timeit
- `σ_c`: visco-plastic characteristic stress
- `σ_c`: visco-plastic exponent
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
    Eᵉ::Tensor{4,3,Float64,81}
    H::Matrix{Float64}
    # store information about slip sytems
    nslipsystems::Int
    slipsystems::Vector{NTuple{2, Vec{3}}}

    function CrystalViscoPlastic(E, ν, σ_y, H_iso, H_kin, q, α_∞, t_star, τ_y, m, slipsystems)
        Eᵉ = elastic_tangent_3D(E, ν)
        nslipsystems = length(slipsystems)
        H = H_iso*(q*ones(nslipsystems, nslipsystems) + (1-q)*diagm(ones(nslipsystems)))

        return new{nslipsystems}(E, ν, τ_y, H_iso, H_kin, q, α_∞, t_star, σ_c, m, Eᵉ, H, nslipsystems, slipsystems)
    end
end

# keyword argument constructor
CrystalViscoPlastic(; E, ν, τ_y, H_iso, H_kin, q, α_∞, t_star, σ_c, m, slipsystems) = CrystalViscoPlastic(E, ν, τ_y, H_iso, H_kin, q, α_∞, t_star, σ_c, m, slipsystems)

# to do: maybe parametrize with nslipsystems and use tuples?
struct CrystalViscoPlasticState{dim,T,M,S} <: AbstractMaterialState
    σ::SymmetricTensor{2,dim,T,M}
    κ::Vector{Float64}
    α::Vector{Float64}
    μ::Vector{Float64}
end

Base.zero(::Type{CrystalViscoPlasticState{dim,T,M,S}}) where {dim,T,M,S} = CrystalViscoPlasticState{dim,T,M,S}(zero(SymmetricTensor{2,dim,T,M}), zeros(T, S), zeros(T, S), zeros(T, S))
initial_material_state(::CrystalViscoPlastic{S}) where S = zero(CrystalViscoPlasticState{3,Float64,6,S})

# constitutive driver operates in 3D, so these can always be 3D
# TODO: Should Residuals have a Type Parameter N for the number of scalar equations?
struct Residuals{CrystalViscoPlastic, T, S}
    σ::SymmetricTensor{2,3,T,6}
    κ::Vector{T}
    α::Vector{T}
    μ::Vector{T}
    function Residuals{CrystalViscoPlastic}(σ::SymmetricTensor{2,3,T,6}, κ::Vector{T}, α::Vector{T}, μ::Vector{T}) where {T}
        S = length(κ)
        length(α) == S || error("$S slipsystems for κ, but $(length(α)) slipsystems for α handed over to Residuals{CrystalViscoPlastic}.")
        length(μ) == S || error("$S slipsystems for κ, but $(length(μ)) slipsystems for μ handed over to Residuals{CrystalViscoPlastic}.")
        new{CrystalViscoPlastic,T,S}(σ, κ, α, μ)
    end
end

get_n_scalar_equations(::Residuals{CrystalViscoPlastic{S}}) where S = 6 + 3S

function Tensors.tomandel!(v::Vector{T}, r::Residuals{CrystalViscoPlastic,T, S}) where {T, S}
    M=6
    # TODO check vector length
    tomandel!(view(v, 1:M), r.σ)
    v[M+1:M+S] = r.κ
    v[M+S+1:M+2S] = r.α
    v[M+2S+1:M+3S] = r.μ
    return v
end