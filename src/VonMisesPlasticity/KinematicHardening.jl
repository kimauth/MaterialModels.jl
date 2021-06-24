# Kinematic hardening
abstract type AbstractKinematicHardening end

"""
    ArmstrongFrederick(Hkin, β∞)

Armstrong-Frederick kinematic hardening law (doi: 10.1179/096034007X207589)

```math
g_{\\mathrm{kin},i}(\\nu, \\boldsymbol{\\beta}_i) = H_\\mathrm{kin} (\\frac{2}{3}\\boldsymbol{\\nu} - \\frac{\\boldsymbol{\\beta}_i}{\\beta_\\infty})
```

# Arguments
- `Hkin`: Kinematic hardening modulus, ``H_\\mathrm{kin}``
- `β∞`: Effective back-stress saturation value, ``\\beta_\\infty``
"""
struct ArmstrongFrederick{T} <: AbstractKinematicHardening
    Hkin::T     # Initial hardening modulus
    β∞::T       # Saturation stress
end
ArmstrongFrederick(;Hkin, β∞) = ArmstrongFrederick(Hkin, β∞)    # Keyword argument constructor

function get_evolution(param::ArmstrongFrederick, ν::SecondOrderTensor, βᵢ::SecondOrderTensor)
    param.Hkin * ((2.0/3.0) * ν - βᵢ/param.β∞)
end

"""
    Delobelle(Hkin, β∞, δ)

Kinematic hardening law according to Delobelle, which combines the Armstrong-Frederick law with the Burlet-Cailletaud law
(doi: 10.1016/S0749-6419(95)00001-1)

```math
g_{\\mathrm{kin},i}(\\nu, \\boldsymbol{\\beta}_i) = H_\\mathrm{kin} \\left[\\frac{2}{3}\\boldsymbol{\\nu} 
                                    - \\delta\\frac{\\boldsymbol{\\beta}_i}{\\beta_\\infty}
                                    - \\frac{2}{3\\beta_\\infty}\\left[1 - \\delta\\right]\\left[\\boldsymbol{\\nu}:\\boldsymbol{\\beta}_i\\right]\\boldsymbol{\\nu}
                                    \\right]
```

# Arguments
- `Hkin`: Kinematic hardening modulus, ``H_\\mathrm{kin}``
- `β∞`: Effective back-stress saturation value, ``\\beta_\\infty``
- `δ`: Amount of Armstrong-Frederick type of kinematic hardening, ``\\delta``

"""
struct Delobelle{T} <: AbstractKinematicHardening
    Hkin::T     # Initial hardening modulus
    β∞::T       # Saturation stress
    δ::T        # Amount of Armstrong-Frederick hardening
end
Delobelle(;Hkin, β∞, δ) = Delobelle(Hkin, β∞, δ)    # Keyword argument constructor

function get_evolution(param::Delobelle, ν::SecondOrderTensor, βᵢ::SecondOrderTensor)
    AF_Term = (param.δ/param.β∞) * βᵢ                        # Armstrong Frederick term
    BC_Term = (2.0/3.0) * (1.0-param.δ)*((ν⊡βᵢ)/param.β∞)*ν  # Burlet Cailletaud term
    return param.Hkin * ((2.0/3.0) * ν - AF_Term - BC_Term)  # Complete evolution 
end


"""
    OhnoWang(Hkin, β∞, m)

Kinematic hardening law according to Ohno-Wang (doi: 10.1016/0749-6419(93)90042-O)

```math
g_{\\mathrm{kin},i}(\\nu, \\boldsymbol{\\beta}_i) = H_\\mathrm{kin} \\left[\\frac{2}{3}\\boldsymbol{\\nu} 
                                    - \\frac{\\boldsymbol{\\beta}_i}{\\beta_\\infty} 
                                    \\frac{\\langle \\boldsymbol{\\nu}:\\boldsymbol{\\beta}_i \\rangle}{\\beta_\\infty}
                                    \\left[\\frac{\\beta_i^\\mathrm{vM}}{\\beta_\\infty}\\right]^m
                                    \\right]
```
where ``\\langle x \\rangle`` is 0 if ``x\\leq 0`` and ``x`` if ``x>0``.
``\\beta_i^\\mathrm{vM} = \\sqrt{2\\boldsymbol{\\beta}_i:\\boldsymbol{\\beta}_i/3}``, noting that
``\\boldsymbol{\\beta}_i`` is deviatoric.

# Arguments
- `Hkin`: Kinematic hardening modulus, ``H_\\mathrm{kin}``
- `β∞`: Effective back-stress saturation value, ``\\beta_\\infty``
- `m`: Exponent in the OhnoWang equation, ``m``

"""
struct OhnoWang{T} <: AbstractKinematicHardening
    Hkin::T     # Initial hardening modulus
    β∞::T       # Saturation stress
    m::T     # Ohno Wang exponent
end
OhnoWang(;Hkin, β∞, m) = OhnoWang(Hkin, β∞, m)    # Keyword argument constructor

function get_evolution(param::OhnoWang{Tp}, ν::SecondOrderTensor, βᵢ::SecondOrderTensor{dim,Tβ}) where{Tp,Tβ,dim}
    β_vm = vonmises_dev(βᵢ)
    if β_vm < param.β∞ * eps(promote_type(Tp,Tβ))
        return param.Hkin * (2.0/3.0) * ν + 0*βᵢ
    end
    mac_term = (macaulay(ν⊡βᵢ) /param.β∞)
    exp_term = (β_vm/param.β∞)^param.m
    return param.Hkin * ((2.0/3.0) * ν - βᵢ * mac_term * exp_term / β_vm )
end