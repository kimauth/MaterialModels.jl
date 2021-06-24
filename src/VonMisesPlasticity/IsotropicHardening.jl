# Isotropic hardening
abstract type AbstractIsotropicHardening end

""" 
    Voce(Hiso, κ∞)

Exponentially saturating isotropic hardening

```math
\\kappa_i = g_{\\mathrm{iso},i}(\\lambda) = \\kappa_\\infty \\left[1 - \\mathrm{exp}\\left(\\frac{H_\\mathrm{iso}}{\\kappa_\\infty} \\lambda \\right)\\right]
```
or alternatively as differential equations
```math
\\dot{\\kappa_i} = \\dot{\\lambda} H_\\mathrm{iso} \\left[1 - \\frac{\\kappa_i}{\\kappa_\\infty}\\right]
```

# Arguments
- `Hiso`: Isotropic hardening modulus, ``H_\\mathrm{iso}``
- `κ∞`: Saturation hardening value, ``\\kappa_\\infty``
    
"""
struct Voce{T} <:AbstractIsotropicHardening
    Hiso::T     # Initial hardening modulus
    κ∞::T       # Saturation stress
end
Voce(;Hiso, κ∞) = Voce(Hiso, κ∞)    # Keyword argument constructor

function get_hardening(param::Voce, λ::Number)
    param.κ∞ * (1.0 - exp(-param.Hiso * λ / param.κ∞))
end

""" 
    Swift(K, λ0, n)

Isotropic hardening by the Swift power law

```math
\\kappa_i = g_{\\mathrm{iso},i}(\\lambda) = K \\left[\\lambda_0 + \\lambda \\right]^n
```

# Arguments
- `K`: ``K``
- `λ0`: ``\\lambda_0``
- `n`: ``n``

"""
struct Swift{T} <:AbstractIsotropicHardening
    K::T
    λ0::T
    n::T 
end
Swift(;K, λ0, n) = Swift(K, λ0, n)    # Keyword argument constructor


function get_hardening(param::Swift, λ::Number)
    param.K * (param.λ0 + λ)^param.n
end

