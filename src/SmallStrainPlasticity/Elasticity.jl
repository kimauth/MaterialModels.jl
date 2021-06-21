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

function material_response(m::LinearIsotropicElasticity, ϵ::SymmetricTensor{2,3}, state_old, Δt=nothing; cache=get_cache(m), options::Dict{Symbol, Any} = Dict{Symbol, Any}())
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