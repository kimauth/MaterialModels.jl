# DiffResults have a bug when being used with MArrays, which is fixed by this specialization
DiffResults.DiffResult(value::MArray, derivs::Tuple{Vararg{MArray}}) = DiffResults.MutableDiffResult(value, derivs)

""" 
    function vonmises(σ::SymmetricTensor{2,3})

Calculate the von Mises effective stress for a symmetric tensor

# Arguments
- `σ::SymmetricTensor{2,3}`: Symmetric stress tensor 

"""
function vonmises(σ::SymmetricTensor{2,3})
    σ_dev = dev(σ)
    return vonmises_dev(σ_dev)
end

""" 
    function vonmises_dev(σ_dev::SymmetricTensor{2,3})

Calculate the von Mises effective stress for a symmetric and deviatoric tensor

# Arguments
- `σ_dev::SymmetricTensor{2,3}`: Symmetric and deviatoric stress tensor 

"""
function vonmises_dev(σ_dev::SymmetricTensor{2,3})
    return sqrt((3.0/2.0) * (σ_dev ⊡ σ_dev))
end

"""
    function macaulay(x)

Calculate the macaulay bracket of x, ``\\langle x \\rangle``
```math
\\langle x \\rangle = \\left\\lbrace \\begin{matrix} 0 & x\\leq 0 \\\\ x & x>0\\end{matrix}
```
"""
macaulay(x::T) where {T} = x > zero(T) ? x : zero(T)