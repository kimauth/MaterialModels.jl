@testset "CrystalViscoPlastic" begin
    slipsystems = MaterialModels.slipsystems(MaterialModels.FCC(), rand(RodriguesParam))
    m = CrystalViscoPlastic(E=200e3, ν=0.3, τ_y=400., H_iso=1e3, H_kin=1e3, q=0.0, α_∞=100., t_star=20., σ_c=50., m=10., slipsystems=slipsystems)
    S = MaterialModels.get_n_slipsystems(m)
    state = initial_material_state(m)

    # strain at yield point for uniaxial stress
    ε11_yield_all = [m.τ_y/ms[1,1]/m.E for ms in m.MS]
    ε11_yield = minimum(abs.(ε11_yield_all))
    i = findfirst(v->v==ε11_yield, abs.(ε11_yield_all))

    # elastic response (yield point)
    Δε = SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? ε11_yield : (i == j ? -ε11_yield*m.ν : 0.0))
    σ, ∂σ∂ε, temp_state = material_response(m, Δε, state)
    @test σ ≈ SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? ε11_yield*m.E : 0.0)
    @test abs(σ ⊡ m.MS[i]) ≈ m.τ_y
    @test ∂σ∂ε ≈ m.Eᵉ
    @test temp_state.κ ≈ zeros(S)
    @test temp_state.α ≈ zeros(S)
    @test temp_state.μ ≈ zeros(S)

    # plastic branch
    Δε = SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? 2ε11_yield : (i == j ? -2ε11_yield*m.ν : 0.0))
    σ, ∂σ∂ε, temp_state = material_response(m, Δε, state)
    @test abs(σ ⊡ m.MS[i]) > m.τ_y
    # these are rather lousy tests, we should come up with something better
    @test !(temp_state.κ[i] ≈ 0.0)
    @test !(temp_state.α[i] ≈ 0.0)
    @test !(temp_state.μ[i] ≈ 0.0)
end

function get_visco_plastic_loading()
    strain1 = range(0.0,  0.01, length=5)
    strain2 = range(0.005, 0.001, length=5)
    strain3 = range(0.001, 0.007, length=5)

    # _C = [strain1..., strain2..., strain3...]
    _C = [strain1[i]-strain1[i-1] for i=2:length(strain1)]
    ε = [SymmetricTensor{2,3}((x, x/10, 0.0, 0.0, 0.0, 0.0)) for x in _C]

    return ε
end  

@testset "CrystalViscoPlastic jld2" begin
    slipsystems = MaterialModels.slipsystems(MaterialModels.FCC(),RodriguesParam(-0.665267, 0.0203875, 1.08633))
    m = CrystalViscoPlastic(E=200e3, ν=0.3, τ_y=400., H_iso=1e3, H_kin=1e3, q=0.0, α_∞=100., t_star=20., σ_c=50., m=10., slipsystems=slipsystems)

    loading = get_visco_plastic_loading()
    # want the same results as from the other implementation
    check_jld2(m, loading, "CrystalViscoPlastic1")#, debug_print=true, OVERWRITE_JLD2=true)
end