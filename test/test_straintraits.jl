@testset "StrainTrait" begin

    mat = NeoHook(λ = 1.0, μ = 1.0)
    state = initial_material_state(mat)
    F = Tensor{2,3}((2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    Δt = 0.0
    I = one(SymmetricTensor{2,3,Float64})


    S, dSdC, state = material_response(MaterialModels.∂S∂C(), mat, F, state, Δt)

    Pᵀ, dPᵀdF, state = material_response(MaterialModels.∂Pᵀ∂F(), mat, F, state, Δt)
    @test Pᵀ == S⋅F'
    @test dPᵀdF == otimesu(F,I) ⊡ dSdC ⊡ otimesu(F',I) + otimesu(S,I)

    S_E, dSdE, state = material_response(MaterialModels.∂S∂E(), mat, F, state, Δt)
    @test S_E == S
    @test 2dSdC == dSdE

end 
