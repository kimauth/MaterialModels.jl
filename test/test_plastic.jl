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

    # @btime constitutive_driver($m, $Δε, $state; cache=$cache, options=$Dict{Symbol,Any}())

    # σ_trial = state.σ + m.Eᵉ ⊡ Δε
    # x0 = MaterialModels.Residuals{Plastic}(σ_trial, state.κ, state.α, state.μ)
    # @btime MaterialModels.residuals($x0, $m, $state, $Δε)
    # @btime frommandel($(MaterialModels.Residuals{Plastic}), $rand(14))
    # x_vector = zeros(14)
    # @btime tomandel!($x_vector, $x0)
    # @btime MaterialModels.vector_residual!($f, $similar(x_vector), $x_vector, m)
    # f(x) = MaterialModels.residuals(x, m, state, Δε)
    # vector_residual!((x->MaterialModels.residuals(x, m, state, zero(SymmetricTensor{2,3}))), r_vector, x_vector, m)
end


function get_Plastic_loading()
    strain1 = range(0.0,  0.005, length=5)
    strain2 = range(0.005, 0.001, length=5)
    strain3 = range(0.001, 0.007, length=5)

    _C = [strain1..., strain2..., strain3...]
    ε = [SymmetricTensor{2,3}((x, x/10, 0.0, 0.0, 0.0, 0.0)) for x in _C]

    return ε
end  

@testset "Plastic checksum" begin
    m = Plastic(E=200e3, ν=0.3, σ_y=200., H=50., r=0.5, κ_∞=13., α_∞=13.)

    loading = get_Plastic_loading()
    check_checksum(m, loading, "Plastic1")#, debug_print=true, OVERWRITE_CHECKSUMS=true)
end

