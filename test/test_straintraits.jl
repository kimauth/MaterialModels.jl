@testset "StrainTrait" begin

    mat = NeoHook(λ = 1.0, μ = 1.0)
    state = initial_material_state(mat)
    F = Tensor{2,3}((2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    Δt = 0.0
    I = one(SymmetricTensor{2,3,Float64})
    C = tdot(F)
    E = (C - one(C))/2

    S, dSdC, state = material_response(MaterialModels.∂S∂C, mat, F, state, Δt)

    Pᵀ, dPᵀdF, state = material_response(MaterialModels.∂Pᵀ∂F, mat, F, state, Δt)
    @test Pᵀ == S⋅F'

    S_E, dSdE, state = material_response(MaterialModels.∂S∂E, mat, F, state, Δt)
    @test S_E == S
    @test 2dSdC == dSdE

    function second_piola_kirchhoff_from_green_lagrange(mat, E)
        C = 2E + one(E)
        S, = material_response(mat, C)
        return S
    end
    dSdE_autodiff = gradient(E->second_piola_kirchhoff_from_green_lagrange(mat, E), E)
    @test dSdE ≈ dSdE_autodiff

    function first_piola_kirchhoff_transposed(mat, F)
        C = tdot(F)
        S, = material_response(mat, C)
        Pᵀ = S ⋅ F' 
        return Pᵀ
    end
    dPᵀdF_autodiff = gradient(F->first_piola_kirchhoff_transposed(mat, F), F)
    @test dPᵀdF ≈ dPᵀdF_autodiff
end 
