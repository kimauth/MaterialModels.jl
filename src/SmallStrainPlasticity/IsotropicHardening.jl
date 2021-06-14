# Isotropic hardening
abstract type AbstractIsotropicHardening end

# Voce type of isotropic hardening (exponentially saturating)
struct IsotropicHardeningVoce{T} <:AbstractIsotropicHardening
    Hiso::T     # Initial hardening modulus
    κ∞::T       # Saturation stress
end
IsotropicHardeningVoce(;Hiso, κ∞) = IsotropicHardeningVoce(Hiso, κ∞)    # Keyword argument constructor

function get_hardening(param::IsotropicHardeningVoce, λ::Number)
    param.κ∞ * (1.0 - exp(-param.Hiso * λ / param.κ∞))
end

# Swift type of kinematic hardening (power law)
struct IsotropicHardeningSwift{T} <:AbstractIsotropicHardening
    K::T
    λ0::T
    n::T 
end
IsotropicHardeningSwift(;K, λ0, n) = IsotropicHardeningSwift(K, λ0, n)    # Keyword argument constructor

function get_hardening(param::IsotropicHardeningSwift, λ::Number)
    param.K * (param.λ0 + λ)^param.n
end

