@testset "StrainTrait" begin

    mat = NeoHook(λ = 1.0, ν = 1.0)
    state = initial_material_state(mat)
    F = Tensor((2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    Δt = 0.0

    S, dSdE, state = material_response(dSdE(), mat, F, state, Δt)

end