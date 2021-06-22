# Linear isotropic elasticity
struct LinearIsotropicElasticity{T} <:AbstractElasticity
    G::T    # Shear modulus
    K::T    # Bulk modulus
end
# Overload initialization method to use more common input parameters
# E: Young's modulus, ν: Poissons ratio
function LinearIsotropicElasticity(E::Number, ν::Number)
    T = promote_type(typeof(E), typeof(ν))
    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)
    return LinearIsotropicElasticity{T}(G, K)
end
LinearIsotropicElasticity(;E, ν) = LinearIsotropicElasticity(E, ν)    # Keyword argument constructor

# Elastic material
get_cache(::LinearIsotropicElasticity) = nothing

initial_material_state(::LinearIsotropicElasticity) = nothing

"""
    material_response(m::LinearIsotropicElasticity, ϵ::SymmetricTensor{2,3}, state_old, Δt=nothing; <keyword arguments>)

Return the stress tensor, stress tangent and the new `MaterialState` 
The total (elastic) strain ϵ is given as input (state_old and Δt can be supplied, but have no influence)
The stress is calculated as

```math
\\boldsymbol{\\sigma} = 2G \\boldsymbol{\\epsilon}^\\mathrm{dev} + 3K \\boldsymbol{\\epsilon}^\\mathrm{vol}
```
where ``\\boldsymbol{\\epsilon}^\\mathrm{vol} = \\boldsymbol{I}:\\boldsymbol{\\epsilon} \\boldsymbol{I}/3.0`` 
is the volumetric strain and ``\\boldsymbol{\\epsilon}^\\mathrm{dev} = \\boldsymbol{\\epsilon} - \\boldsymbol{\\epsilon}^\\mathrm{vol}``
is the deviatoric strain. 

# Keyword arguments
- `cache`: Not used
- `options::Dict{Symbol, Any}`: Not used
"""
function material_response(m::LinearIsotropicElasticity, ϵ::SymmetricTensor{2,3}, state_old=initial_material_state(m), Δt=nothing; cache=get_cache(m), options::Dict{Symbol, Any} = Dict{Symbol, Any}())
    ν = (3*m.K - 2*m.G)/(2*(3*m.K+m.G))    # Calculate poissons ratio
    
    σ = 2 * m.G*dev(ϵ) + 3 * m.K*vol(ϵ)   # Calculate stress
    
    # Create stiffness matrix
    δ(i,j) = i == j ? 1.0 : 0.0 # helper function
    Dfun(i,j,k,l) = 2.0*m.G *( 0.5*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k)) + ν/(1.0-2.0ν)*δ(i,j)*δ(k,l))
    𝔻 = SymmetricTensor{4, 3}(Dfun)
    
    # Return updated values
    state = state_old
    return σ, 𝔻, state
end