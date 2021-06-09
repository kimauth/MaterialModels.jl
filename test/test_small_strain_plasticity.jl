@testset "SmallStrainPlasticity" begin
    # constructor
    m = Chaboche(elastic=Elastic(E=210.e3, ν=0.3),
                 σ_y0=100.0,
                 isotropic=(Iso_Voce(Hiso=100000.0, κ∞=100.0),), # Can add more dragstresses by more entries in Tuple)
                 kinematic=(Kin_AF(Hkin=1000000.0, β∞=200.0),)   # Can add more backstresses by more entries in Tuple
    )
    cache = get_cache(m)

    # initial state (not used here)
    state = initial_material_state(m)
    
    # strain at yield point for uniaxial stress some test case at some plastic strain:
    ϵ = SymmetricTensor{2, 3}((i,j) -> i==j ? (i==1 ? 0.004 : -0.000782017) : 0.0)

    ϵₚ_old = SymmetricTensor{2, 3}((i,j) -> i==j ? (i==1 ? 0.00153388 : -0.000766938) : 0.0)
    λ_old = 0.0015338757291717328
    β_old = SymmetricTensor{2, 3}((i,j) -> i==j ? (i==1 ? 133.268 : -66.6339) : 0.0)

    state_old = MaterialModels.ChabocheState(ϵₚ_old,λ_old, (β_old,))

    Δt = 1.0    # No influence...
    
    σ, ∂σ∂ε, temp_state, converged = material_response(m, ϵ, state, Δt; cache=cache)

    @test converged

end