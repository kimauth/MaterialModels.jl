@testset "StrainTrait" begin

    mat = NeoHook(λ = 1.0, μ = 1.0)
    state = initial_material_state(mat)
    F = Tensor{2,3}((2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    Δt = 0.0

    S, _dSdE, state = material_response(dSdE(), mat, F, state, Δt)
    Pᵀ, _dPᵀdF, state = material_response(dPᵀdF(), mat, F, state, Δt)

end