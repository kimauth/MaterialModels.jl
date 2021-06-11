# Isotropic hardening
abstract type AbstractIsoHard end

# Voce type of isotropic hardening (exponentially saturating)
struct Iso_Voce{T} <:AbstractIsoHard
    Hiso::T     # Initial hardening modulus
    κ∞::T       # Saturation stress
end
Iso_Voce(;Hiso, κ∞) = Iso_Voce(Hiso, κ∞)    # Keyword argument constructor

function IsotropicHardening(param::Iso_Voce, λ::Number)
    param.κ∞ * (1.0 - exp(-param.Hiso * λ / param.κ∞))
end

# Swift type of kinematic hardening (power law)
struct Iso_Swift{T} <:AbstractIsoHard
    K::T
    λ0::T
    n::T 
end
Iso_Swift(;K, λ0, n) = Iso_Swift(K, λ0, n)    # Keyword argument constructor

function IsotropicHardening(param::Iso_Swift, λ::Number)
    param.K * (param.λ0 + λ)^n
end

