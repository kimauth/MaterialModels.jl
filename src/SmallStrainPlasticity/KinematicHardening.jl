using Tensors

# Kinematic hardening
abstract type AbstractKinHard{T} end
struct Kin_AF{T} <: AbstractKinHard{T}
    Hkin::T     # Initial hardening modulus
    β∞::T       # Saturation stress
end
function KinematicEvolution(param::Kin_AF{Tp}, 𝛎::SecondOrderTensor{dim,Tν}, 𝛃ᵢ::SecondOrderTensor{dim,Tβ}) where{Tp,Tν,Tβ,dim}
    param.Hkin * ((2.0/3.0) * 𝛎 - 𝛃ᵢ/param.β∞)
end

struct Kin_DB{T} <: AbstractKinHard{T}
    Hkin::T     # Initial hardening modulus
    β∞::T       # Saturation stress
    δ::T        # Amount of Armstrong-Frederick hardening
end
function KinematicEvolution(param::Kin_DB{Tp}, 𝛎::SecondOrderTensor{dim,Tν}, 𝛃ᵢ::SecondOrderTensor{dim,Tβ}) where{Tp,Tν,Tβ,dim}
    AF_Term = (param.δ/param.β∞) * 𝛃ᵢ                        # Armstrong Frederick term
    BC_Term = (2.0/3.0) * (1.0-param.δ)*((ν⊡𝛃ᵢ)/param.β∞)*ν  # Burlet Cailletaud term
    return param.Hkin * ((2.0/3.0) * 𝛎 - AF_Term - BC_Term)  # Complete evolution 
end

struct Kin_OW{T} <: AbstractKinHard{T}
    Hkin::T     # Initial hardening modulus
    β∞::T       # Saturation stress
    mexp::T     # Ohno Wang exponent
end
function KinematicEvolution(param::Kin_OW{Tp}, 𝛎::SecondOrderTensor{dim,Tν}, 𝛃ᵢ::SecondOrderTensor{dim,Tβ}) where{Tp,Tν,Tβ,dim}
    β_vm = vonMisesDev(𝛃ᵢ)
    if β_vm < param.β∞ * eps(T)
        return param.Hkin * (2.0/3.0) * 𝛎 + 0*𝛃ᵢ
    end
    mac_term = (macaulay(𝛎⊡𝛃ᵢ) /param.β∞)
    exp_term = (β_vm/param.β∞)^param.mexp 
    return param.Hkin * ((2.0/3.0) * 𝛎 - 𝛃ᵢ * mac_term * exp_term / β_vm )
end