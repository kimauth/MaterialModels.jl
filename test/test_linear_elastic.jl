# constructor

@testset "LinearElastic" begin
    m = LinearElastic(E=200e3, ν=0.3)

    # initial state
    state = initial_material_state(m)

    test_cache = get_cache(m)
    #@test isnothing(test_cache)

    # constitutive driver
    ε = rand(SymmetricTensor{2,3})
    σ, ∂σ∂ε, temp_state = material_response(m, ε, state)
    @test σ == m.Eᵉ ⊡ ε
    @test ∂σ∂ε == m.Eᵉ

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