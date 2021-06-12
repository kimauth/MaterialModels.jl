@testset "Plastic" begin
    # constructor
    m = Plastic(E=200e3, ν=0.3, σ_y=200., H=50., r=0.5, κ_∞=13., α_∞=13.)
    cache = get_cache(m)

    # initial state
    state = initial_material_state(m)
    @test state.σ == zero(SymmetricTensor{2,3})

    # strain at yield point for uniaxial stress
    ε11_yield = m.σ_y / m.E

    # elastic branch
    Δε = SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? 0.5ε11_yield : (i == j ? -0.5ε11_yield*m.ν : 0.0))
    σ, ∂σ∂ε, temp_state = material_response(m, Δε, state; cache=cache)
    @test σ ≈ SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? 0.5ε11_yield*m.E : 0.0)
    @test ∂σ∂ε == m.Eᵉ
    @test temp_state.κ == 0.0
    @test temp_state.α == zero(SymmetricTensor{2,3})
    @test temp_state.μ == 0.0

    # yield point
    Δε = Δε = SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? ε11_yield : (i == j ? -ε11_yield*m.ν : 0.0))
    σ, ∂σ∂ε, temp_state = material_response(m, Δε, state; cache=cache)
    @test sqrt(3/2)*norm(dev(σ-temp_state.α)) ≈ m.σ_y

    # plastic branch
    Δε = Δε = SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? 2ε11_yield : (i == j ? -2ε11_yield*m.ν : 0.0))
    σ, ∂σ∂ε, temp_state = material_response(m, Δε, state; cache=cache)
    @test sqrt(3/2)*norm(dev(σ-temp_state.α)) > m.σ_y
end


function get_Plastic_loading()
    strain1 = range(0.0,  0.005, length=5)
    strain2 = range(0.005, 0.001, length=5)
    strain3 = range(0.001, 0.007, length=5)

    _C = [strain1..., strain2..., strain3...]
    ε = [SymmetricTensor{2,3}((x, x/10, 0.0, 0.0, 0.0, 0.0)) for x in _C]

    return ε
end  

@testset "Plastic jld2" begin
    m = Plastic(E=200e3, ν=0.3, σ_y=200., H=50., r=0.5, κ_∞=13., α_∞=13.)

    loading = get_Plastic_loading()
    check_jld2(m, loading, "Plastic1")#, debug_print=true, OVERWRITE_JLD2=true)
end

