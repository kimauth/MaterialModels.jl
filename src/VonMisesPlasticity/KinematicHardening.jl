# Kinematic hardening
abstract type AbstractKinematicHardening end

# Armstrong-Frederick 
struct ArmstrongFrederick{T} <: AbstractKinematicHardening
    Hkin::T     # Initial hardening modulus
    β∞::T       # Saturation stress
end
ArmstrongFrederick(;Hkin, β∞) = ArmstrongFrederick(Hkin, β∞)    # Keyword argument constructor

"""
    get_evolution(param::ArmstrongFrederick, 𝛎::SecondOrderTensor, 𝛃ᵢ::SecondOrderTensor)

    Armstrong-Frederick kinematic hardening law (doi: 10.1179/096034007X207589)

    ```math
    g_{\\mathrm{kin},i}(\\nu, \\beta_i) = Hkin (\\frac{2}{3}\\boldsymbol{\\nu} - \\frac{\\boldsymbol{\\beta}_i}{\\beta_\\infty})
    ```
"""
function get_evolution(param::ArmstrongFrederick, 𝛎::SecondOrderTensor, 𝛃ᵢ::SecondOrderTensor)
    param.Hkin * ((2.0/3.0) * 𝛎 - 𝛃ᵢ/param.β∞)
end

# Delobelle (Combination of Armstrong-Frederick and Burlet-Cailletaud)
struct Delobelle{T} <: AbstractKinematicHardening
    Hkin::T     # Initial hardening modulus
    β∞::T       # Saturation stress
    δ::T        # Amount of Armstrong-Frederick hardening
end
Delobelle(;Hkin, β∞, δ) = Delobelle(Hkin, β∞, δ)    # Keyword argument constructor

"""
    get_evolution(param::Delobelle, 𝛎::SecondOrderTensor, 𝛃ᵢ::SecondOrderTensor)

    Kinematic hardening law according to Delobelle, which combines the Armstrong-Frederick law with the Burlet-Cailletaud law
    (doi: 10.1016/S0749-6419(95)00001-1)

    ```math
    g_{\\mathrm{kin},i}(\\nu, \\beta_i) = Hkin \\left[\\frac{2}{3}\\boldsymbol{\\nu} 
                                        - \\delta\\frac{\\boldsymbol{\\beta}_i}{\\beta_\\infty}
                                        - \\frac{2}{3\\beta_\\infty}\\left[1 - \\delta\\right]\\left[\\boldsymbol{\\nu}:\\boldsymbol{\\beta}_i\right]\\boldsymbol{\\nu}
                                        \\right]
    ```
    
"""
function get_evolution(param::Delobelle, 𝛎::SecondOrderTensor, 𝛃ᵢ::SecondOrderTensor)
    AF_Term = (param.δ/param.β∞) * 𝛃ᵢ                        # Armstrong Frederick term
    BC_Term = (2.0/3.0) * (1.0-param.δ)*((ν⊡𝛃ᵢ)/param.β∞)*ν  # Burlet Cailletaud term
    return param.Hkin * ((2.0/3.0) * 𝛎 - AF_Term - BC_Term)  # Complete evolution 
end

# Ohno-Wang
struct OhnoWang{T} <: AbstractKinematicHardening
    Hkin::T     # Initial hardening modulus
    β∞::T       # Saturation stress
    mexp::T     # Ohno Wang exponent
end
OhnoWang(;Hkin, β∞, mexp) = OhnoWang(Hkin, β∞, mexp)    # Keyword argument constructor

""" 
    get_evolution(param::OhnoWang{Tp}, 𝛎::SecondOrderTensor, 𝛃ᵢ::SecondOrderTensor{dim,Tβ}) where{Tp,Tβ,dim}

    Kinematic hardening law according to Ohno-Wang (doi: 10.1016/0749-6419(93)90042-O)

    ```math
    g_{\\mathrm{kin},i}(\\nu, \\beta_i) = Hkin \\left[\\frac{2}{3}\\boldsymbol{\\nu} 
                                        - \\frac{\\boldsymbol{\\beta}_i}{\\beta_\\infty} 
                                        \\frac{\\langle \\boldsymbol{\\nu}:\\boldsymbol{\\beta}_i \\rangle}{\\beta_\\infty}
                                        \\left[\\frac{\\beta_\\mathrm{vM}}{\\beta_\\infty}\right]^\\mathrm{mexp}
                                        \\right]
    ```
    
"""
function get_evolution(param::OhnoWang{Tp}, 𝛎::SecondOrderTensor, 𝛃ᵢ::SecondOrderTensor{dim,Tβ}) where{Tp,Tβ,dim}
    β_vm = vonmises_dev(𝛃ᵢ)
    if β_vm < param.β∞ * eps(promote_type(Tp,Tβ))
        return param.Hkin * (2.0/3.0) * 𝛎 + 0*𝛃ᵢ
    end
    mac_term = (macaulay(𝛎⊡𝛃ᵢ) /param.β∞)
    exp_term = (β_vm/param.β∞)^param.mexp 
    return param.Hkin * ((2.0/3.0) * 𝛎 - 𝛃ᵢ * mac_term * exp_term / β_vm )
end