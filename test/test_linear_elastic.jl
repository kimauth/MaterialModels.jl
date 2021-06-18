# constructor

@testset "LinearElastic" begin
    m = LinearElastic(E=200e3, ν=0.3)

    # initial state
    state = initial_material_state(m)
    @test state.σ == zero(SymmetricTensor{2,3})

    # constitutive driver
    Δε = rand(SymmetricTensor{2,3})
    σ, ∂σ∂ε, temp_state = material_response(m, Δε, state)
    @test σ == temp_state.σ
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