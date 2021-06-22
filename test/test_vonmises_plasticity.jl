@testset "VonMisesPlasticity" begin
    # Basic setup with Voce isotropic hardening and one back-stress of Armstrong-Frederick time

    # constructor
    m = VonMisesPlasticity(elastic=LinearElastic(E=210.e3, ν=0.3),
                           σ_y0=100.0,
                           isotropic=(Voce(Hiso=100000.0, κ∞=100.0),), # Can add more dragstresses by more entries in Tuple)
                           kinematic=(ArmstrongFrederick(Hkin=1000000.0, β∞=200.0),)   # Can add more backstresses by more entries in Tuple
    )
    cache = get_cache(m)

    # initial state (not used here)
    state = initial_material_state(m)
    
    # strain at yield point for uniaxial stress some test case at some plastic strain:
    ϵ = SymmetricTensor{2, 3}((i,j) -> i==j ? (i==1 ? 0.004 : -0.000782017) : 0.0)

    ϵₚ_old = SymmetricTensor{2, 3}((i,j) -> i==j ? (i==1 ? 0.00153388 : -0.000766938) : 0.0)
    λ_old = 0.0015338757291717328
    β_old = SymmetricTensor{2, 3}((i,j) -> i==j ? (i==1 ? 133.268 : -66.6339) : 0.0)

    state_old = MaterialModels.VonMisesPlasticityState(ϵₚ_old,λ_old, (β_old,))

    Δt = 1.0    # No influence...
    
    σ, ∂σ∂ε, temp_state = material_response(m, ϵ, state, Δt; cache=cache)

    @test true  # Check that it ran (throws error if not converged)

    # Example with a more advanced material:
    # Linear isotropic elasticity 
    # Two isotropic hardening laws: Voce and Swift 
    # Two back-stresses, one Armstrong-Frederick and one Ohno-Wang
    m = VonMisesPlasticity(elastic=LinearElastic(E=210.e3, ν=0.3),
                           σ_y0=100.0,
                           isotropic=(Voce(Hiso=100000.0, κ∞=100.0),
                                      Swift(K=100.0, λ0=1.0e-2, n=0.5)),
                           kinematic=(ArmstrongFrederick(Hkin=40.e3, β∞=200.0),
                                      OhnoWang(Hkin=30.e3, β∞=200.0, mexp=4.0))
    )

    cache = get_cache(m)
    state = initial_material_state(m)
    Δt = 1.0    # No influence...
    ϵ = SymmetricTensor{2, 3}((i,j) -> i==j ? (i==1 ? 0.1/100.0 : -0.3*0.1/100.0) : 0.0)
    σ, ∂σ∂ε, temp_state = material_response(m, ϵ, state, Δt; cache=cache)

    @test true  # Check that it ran (throws error if not converged)

end