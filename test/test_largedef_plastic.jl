@testset "MatHyperElasticPlastic" begin
    # constructor
    E=200e3
    ν=0.3
    neohook = NeoHook(
        μ = E / (2(1 + ν)),
        λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    )

    m = MatHyperElasticPlastic(
        elastic_material = neohook,
        σ_y = 200.0, 
        H = 50.0,
    )

    # initial state
    state = initial_material_state(m)

    # elastic branch
    ε11_yield = m.σ_y / E 
    Δt = 0.0

    F = Tensor{2,3}((0.5ε11_yield + 1, 0.0, 0.0, 0.0, (-0.5ε11_yield*ν + 1.0), 0.0, 0.0, 0.0, (-0.5ε11_yield*ν + 1.0)))
    C = tdot(F)
    S_neo, ∂S_neo, _ = material_response(neohook, C)
    S, ∂S, temp_state = material_response(m, C, state)
    _,_,_,extras = material_response(m, C, state, Δt , :extras)

    @test S_neo ≈ S
    @test ∂S == ∂S_neo
    @test temp_state.ϵᵖ == 0.0
    @test temp_state.Fᵖ == one(Tensor{2,3,Float64})
    @test extras.D == 0.0 # No dissipation

    # Plastic branch
    F = Tensor{2,3}((1.1ε11_yield + 1, 0.0, 0.0, 0.0, (-1.1ε11_yield*ν + 1.0), 0.0, 0.0, 0.0, (-1.1ε11_yield*ν + 1.0)))
    C = tdot(F)

    S, ∂S, temp_state, extras = material_response(m, C, state, Δt, :extras)

    @test temp_state.ϵᵖ > 0.0
    @test extras.D > 0.0

end


function get_MatHyperElasticPlastic_loading()
    strain1 = range(0.0,  0.002, length=5)
    strain2 = range(0.002, 0.001, length=5)
    strain3 = range(0.001, 0.007, length=5)

    _x = [strain1[1:end]..., strain2[1:end]..., strain3[1:end]...]
    _F = [Tensor{2,3}((x + 1.0, x/10, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)) for x in _x]
    C = [tdot(F) for F in _F]

    return C
end  

@testset "MatHyperElasticPlastic jld2" begin

    E=200e3
    ν=0.3
    m = MatHyperElasticPlastic(
        elastic_material = MaterialModels.NeoHook(
            μ = E / (2(1 + ν)),
            λ = (E * ν) / ((1 + ν) * (1 - 2ν))
        ),
        σ_y = 200.0, 
        H = 50.0,
    )
    loading = get_MatHyperElasticPlastic_loading()
    check_jld2(m, loading, "MatHyperElasticPlastic1")#, debug_print=true, OVERWRITE_JLD2=true)
end