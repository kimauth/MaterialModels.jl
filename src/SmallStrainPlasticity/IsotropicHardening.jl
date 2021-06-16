# Isotropic hardening
abstract type AbstractIsotropicHardening end

# Voce type of isotropic hardening (exponentially saturating)
struct Voce{T} <:AbstractIsotropicHardening
    Hiso::T     # Initial hardening modulus
    κ∞::T       # Saturation stress
end
Voce(;Hiso, κ∞) = Voce(Hiso, κ∞)    # Keyword argument constructor

function get_hardening(param::Voce, λ::Number)
    param.κ∞ * (1.0 - exp(-param.Hiso * λ / param.κ∞))
end

# Swift type of kinematic hardening (power law)
struct Swift{T} <:AbstractIsotropicHardening
    K::T
    λ0::T
    n::T 
end
Swift(;K, λ0, n) = Swift(K, λ0, n)    # Keyword argument constructor

function get_hardening(param::Swift, λ::Number)
    param.K * (param.λ0 + λ)^param.n
end

