

"""
    TransverselyIsotropic

Transversely isotropic elasticity.

The material direction (the vector normal to the symmetry plane) is specified in `TransverselyIsotropicState`,
which is constructed with `initial_material_state(::TransverselyIsotropic, direction::Vec{3})`. The
default direction is [1.0, 0.0, 0.0].

# Arguments
- `ν::Float64`: Poisson's ratio
- `E_L::Float64`: LongitudinalYoung's modulus
- `E_T::Float64`: Transverse Young's modulus
- `G_LT::Float64`: In-plane shear modulus
- `ν_LT::Float64`: In-plane Poisson's ratio
- `ν_TT::Float64`: Out-of-plane Poisson's ratio
"""
struct TransverselyIsotropic <: MaterialModels.AbstractMaterial
    L⊥::Float64
    L₌::Float64
    M₌::Float64
    G⊥::Float64
    G₌::Float64
    #
    I   ::SymmetricTensor{2,3,Float64,6}
    Iˢʸᵐ::SymmetricTensor{4,3,Float64,36}
    IoI ::SymmetricTensor{4,3,Float64,36} 

    function TransverselyIsotropic(L⊥::Float64, L₌::Float64, M₌::Float64, G⊥::Float64, G₌::Float64)
        I = one(SymmetricTensor{2,3})
        Iˢʸᵐ = 0.5 * symmetric( (otimesu(I,I) + otimesl(I,I)) )
        IoI = symmetric(I ⊗ I)
        new(L⊥, L₌, M₌, G⊥, G₌, I, Iˢʸᵐ, IoI)
    end
end

function TransverselyIsotropic(; 
    E_L::T, 
    E_T::T, 
    G_LT::T,
    ν_LT::T, 
    ν_TT::T) where T

    M₌ = (E_L^2*(ν_TT - 1))/(2*E_T*ν_LT^2 - E_L + E_L*ν_TT)
    L₌ = -(E_L*E_T*ν_LT)/(2*E_T*ν_LT^2 - E_L + E_L*ν_TT)
    L⊥ = -(E_T*(E_T*ν_LT^2 + E_L*ν_TT))/((ν_TT + 1)*(2*E_T*ν_LT^2 - E_L + E_L*ν_TT))
    G⊥ = E_T/(2*(ν_TT + 1))
    G₌ = G_LT

    return TransverselyIsotropic(L⊥, L₌, M₌, G⊥, G₌)
end

struct TransverselyIsotropicState <: MaterialModels.AbstractMaterialState
    a3::Vec{3,Float64}
end

function initial_material_state(::TransverselyIsotropic, a3::Vec{3,Float64} = Vec{3,Float64}((1.0, 0.0, 0.0)))
    return TransverselyIsotropicState(a3)
end

"""
    material_response(m::TransverselyIsotropic, ε::SymmetricTensor{2,3}, state::TransverselyIsotropicState)

Return the stress tensor and the stress tangent for the given strain ε such that

```math
\\boldsymbol{\\sigma} = \\mathbf{E} : \\boldsymbol{\\varepsilon}
```

where

```math
\\mathbf{E} = L_\\perp \\boldsymbol{I} \\otimes \\boldsymbol{I} + [L_\\parallel - L_\\perp][\\boldsymbol{I}\\otimes\\boldsymbol{A} + \\boldsymbol{A}\\otimes\\boldsymbol{I}]
+ [M_\\parallel - 4G_\\parallel + 2G_\\perp - 2L_\\parallel + L_\\perp]\\boldsymbol{A}\\otimes\\boldsymbol{A} + 4[G_\\parallel-G_\\perp]\\mathbf{A}
```

```math
\\mathbf{A} = \\frac{1}{4} (\\boldsymbol{A} \\overline{\\otimes} \\boldsymbol{I} + \\boldsymbol{A} \\underline{\\otimes} \\boldsymbol{I} + \\boldsymbol{I} \\overline{\\otimes} \\boldsymbol{A} + \\boldsymbol{I} \\underline{\\otimes} \\boldsymbol{A}) 
```

```math
\\boldsymbol{A} = \\boldsymbol{a} \\otimes \\boldsymbol{a}
```

and where \$\\boldsymbol{a}\$ is the vector normal to the plane of symmetry.

"""
function material_response(m::TransverselyIsotropic, ε::SymmetricTensor{2,3}, state::TransverselyIsotropicState, Δt=nothing; cache=nothing, options=nothing)
    a3 = state.a3
    I = m.I
    A = symmetric( a3 ⊗ a3 )
    𝔸 = 0.25 * symmetric( otimesu(A,I) + otimesl(A,I) + otimesu(I,A) + otimesl(I,A) )
    
    E = m.L⊥                                  * m.IoI                + 
        2m.G⊥                                 * m.Iˢʸᵐ               + 
        (m.L₌ - m.L⊥)                         * symmetric(I⊗A + A⊗I) + 
        (m.M₌ - 4m.G₌ + 2m.G⊥ - 2m.L₌ + m.L⊥) * symmetric(A ⊗ A)     + 
        4(m.G₌ - m.G⊥)                        * 𝔸

    σ = E ⊡ ε

    return σ, E, state
end