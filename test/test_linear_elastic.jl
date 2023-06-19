# constructor

@testset "LinearElastic" begin
    E=200e3
    ν=0.3
    m = LinearElastic(; E, ν)

    # initial state
    state = initial_material_state(m)

    test_cache = get_cache(m)
    @test isnothing(test_cache)

    # constitutive driver
    ε = rand(SymmetricTensor{2,3})
    σ, ∂σ∂ε, temp_state = material_response(m, ε, state)
    @test σ == m.Eᵉ ⊡ ε
    @test ∂σ∂ε == m.Eᵉ

    #Compare with voigt notation from https://en.wikipedia.org/wiki/Hooke%27s_law
    _C = [   1        -ν       -ν        0       0         0;
             -ν        1       -ν        0       0         0;
             -ν        -ν       1        0       0         0;
             0         0        0        2+2ν  0           0;
             0         0        0        0       2+2ν      0;
             0         0        0        0       0         2+2ν] * (1/E)

    C = tovoigt(inv(∂σ∂ε), offdiagscale=2.0)
    @test _C ≈ C

    # stress/ strain measures for compatibility with finite strain system
    @test MaterialModels.native_strain_type(LinearElastic) == SmallStrain
    @test MaterialModels.native_stress_type(LinearElastic) == MaterialModels.TrueStress

end

function get_LinearElastic_loading()
    loading = range(0.0,  0.02, length=2)
    return [SymmetricTensor{2,3}((ε, ε/10, 0.0, 0.0, 0.0, 0.0)) for ε in loading]
end  

@testset "LinearElastic jld2" begin
    m = LinearElastic(E=200e3, ν=0.3)
    loading = get_LinearElastic_loading()
    check_jld2(m, loading, "LinearElastic1")#, OVERWRITE_JLD2=false)
end
