# Isotropic hardening
abstract type AbstractIsoHard{T} end

struct Iso_Voce{T} <:AbstractIsoHard{T}
    Hiso::T     # Initial hardening modulus
    κ∞::T       # Saturation stress
end
function IsotropicHardening(param::Iso_Voce, λ::Number)
    param.κ∞ * (1.0 - exp(-param.Hiso * λ / param.κ∞))
end

struct Iso_Swift{T} <:AbstractIsoHard{T}
    K::T
    λ0::T
    n::T 
end
function IsotropicHardening(param::Iso_Swift, λ::Number)
    param.K * (param.λ0 + λ)^n
end

