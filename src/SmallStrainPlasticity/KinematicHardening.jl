using Tensors

# Kinematic hardening
abstract type AbstractKinHard{T} end

# Armstrong-Frederick 
struct Kin_AF{T} <: AbstractKinHard{T}
    Hkin::T     # Initial hardening modulus
    β∞::T       # Saturation stress
end
Kin_AF(;Hkin, β∞) = Kin_AF(Hkin, β∞)    # Keyword argument constructor

"""
    KinematicEvolution(param::Kin_AF, 𝛎::SecondOrderTensor, 𝛃ᵢ::SecondOrderTensor)

    Armstrong-Frederick kinematic hardening law

    ```math
    g_{\\mathrm{kin},i}(\\nu, \\beta_i) = Hkin (\\frac{2}{3}\\boldsymbol{\\nu} - \\frac{\\boldsymbol{\\beta}_i}{\\beta_\\infty})
    ```
"""
function KinematicEvolution(param::Kin_AF, 𝛎::SecondOrderTensor, 𝛃ᵢ::SecondOrderTensor)
    param.Hkin * ((2.0/3.0) * 𝛎 - 𝛃ᵢ/param.β∞)
end

# Delobelle (Combination of Armstrong-Frederick and Burlet-Cailletaud)
struct Kin_DB{T} <: AbstractKinHard{T}
    Hkin::T     # Initial hardening modulus
    β∞::T       # Saturation stress
    δ::T        # Amount of Armstrong-Frederick hardening
end
Kin_DB(;Hkin, β∞, δ) = Kin_DB(Hkin, β∞, δ)    # Keyword argument constructor

"""
    KinematicEvolution(param::Kin_DB, 𝛎::SecondOrderTensor, 𝛃ᵢ::SecondOrderTensor)

    Kinematic hardening law according to Delobelle, which combines the Armstrong-Frederick law with the Burlet-Cailletaud law

    ```math
    g_{\\mathrm{kin},i}(\\nu, \\beta_i) = Hkin \\left[\\frac{2}{3}\\boldsymbol{\\nu} 
                                        - \\delta\\frac{\\boldsymbol{\\beta}_i}{\\beta_\\infty}
                                        - \\frac{2}{3\\beta_\\infty}\\left[1 - \\delta\\right]\\left[\\boldsymbol{\\nu}:\\boldsymbol{\\beta}_i\right]\\boldsymbol{\\nu}
                                        \\right]
    ```
    
"""
function KinematicEvolution(param::Kin_DB, 𝛎::SecondOrderTensor, 𝛃ᵢ::SecondOrderTensor)
    AF_Term = (param.δ/param.β∞) * 𝛃ᵢ                        # Armstrong Frederick term
    BC_Term = (2.0/3.0) * (1.0-param.δ)*((ν⊡𝛃ᵢ)/param.β∞)*ν  # Burlet Cailletaud term
    return param.Hkin * ((2.0/3.0) * 𝛎 - AF_Term - BC_Term)  # Complete evolution 
end

# Ohno-Wang
struct Kin_OW{T} <: AbstractKinHard{T}
    Hkin::T     # Initial hardening modulus
    β∞::T       # Saturation stress
    mexp::T     # Ohno Wang exponent
end
Kin_OW(;Hkin, β∞, mexp) = Kin_OW(Hkin, β∞, mexp)    # Keyword argument constructor

""" 
    KinematicEvolution(param::Kin_OW{Tp}, 𝛎::SecondOrderTensor, 𝛃ᵢ::SecondOrderTensor{dim,Tβ}) where{Tp,Tβ,dim}

    Kinematic hardening law according to Ohno-Wang

    ```math
    g_{\\mathrm{kin},i}(\\nu, \\beta_i) = Hkin \\left[\\frac{2}{3}\\boldsymbol{\\nu} 
                                        - \\frac{\\boldsymbol{\\beta}_i}{\\beta_\\infty} 
                                        \\frac{\\langle \\boldsymbol{\\nu}:\\boldsymbol{\\beta}_i \\rangle}{\\beta_\\infty}
                                        \\left[\\frac{\\beta_\\mathrm{vM}}{\\beta_\\infty}\right]^\\mathrm{mexp}
                                        \\right]
    ```
    
"""
function KinematicEvolution(param::Kin_OW{Tp}, 𝛎::SecondOrderTensor, 𝛃ᵢ::SecondOrderTensor{dim,Tβ}) where{Tp,Tβ,dim}
    β_vm = vonMisesDev(𝛃ᵢ)
    if β_vm < param.β∞ * eps(promote_type(Tp,Tβ))
        return param.Hkin * (2.0/3.0) * 𝛎 + 0*𝛃ᵢ
    end
    mac_term = (macaulay(𝛎⊡𝛃ᵢ) /param.β∞)
    exp_term = (β_vm/param.β∞)^param.mexp 
    return param.Hkin * ((2.0/3.0) * 𝛎 - 𝛃ᵢ * mac_term * exp_term / β_vm )
end