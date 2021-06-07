using Tensors

# Elasticity
struct Elastic{T}
    G::T    # Shear modulus
    K::T    # Bulk modulus
end
# Overload initialization method to use more common input parameters
# E: Young's modulus, ν: Poissons ratio
function Elastic(E::Number, ν::Number)
    T = promote_type(typeof(E), typeof(ν))
    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)
    return Elastic{T}(G, K)
end

# Elastic material
function material_model(cache, material::Elastic, ϵ::T2, state_old, Δt::AbstractFloat) where{T2<:SymmetricTensor{2,3,T}} where T
    ν = (3material.K - 2material.G)/(2*(3material.K+material.G))    # Calculate poissons ratio
    
    σ = 2 * material.G*dev(ϵ) + 3 * material.K*vol(ϵ)   # Calculate stress
    
    # Create stiffness matrix
    δ(i,j) = i == j ? 1.0 : 0.0 # helper function
    Dfun(i,j,k,l) = 2.0*material.G *( 0.5*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k)) + ν/(1.0-2.0ν)*δ(i,j)*δ(k,l))
    𝔻 = SymmetricTensor{4, 3}(Dfun)
    
    # Return updated values
    converged = true
    state = state_old
    return σ, 𝔻, state, converged
end