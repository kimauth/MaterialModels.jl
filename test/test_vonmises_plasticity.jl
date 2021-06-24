# Functions required as Plastic use the strain increment
get_ramp_value(::Plastic, val_init, val_max, i, num_steps) = (val_max-val_init)/num_steps
get_ramp_value(::VonMisesPlasticity, val_init, val_max, i, num_steps) = val_init + (val_max-val_init)*i/num_steps

function uniaxial_loading(m, ϵ11_init, ϵ11_max, num_steps, t_max, options=Dict{Symbol, Any}())
    state = initial_material_state(m)
    cache = get_cache(m)
    dim = UniaxialStress()
    ϵ = SymmetricTensor{2,1}((ϵ11_init,))
    σ, dσdϵ, state = material_response(dim, m, ϵ, state, t_max/num_steps; cache=cache, options=options)
    for i=1:num_steps
        ϵ = SymmetricTensor{2,1}((get_ramp_value(m, ϵ11_init, ϵ11_max, i, num_steps),))
        σ, dσdϵ, state = material_response(dim, m, ϵ, state, t_max/num_steps; cache=cache, options=options)
    end
    return σ, dσdϵ, state
end

function shear_loading(m, ϵ21_init, ϵ21_max, num_steps, t_max, options=Dict{Symbol, Any}())
    state = initial_material_state(m)
    cache = get_cache(m)
    ϵ = SymmetricTensor{2,3}((i,j)-> i==2 && j==1 ? ϵ21_init : zero(typeof(ϵ21_init)))
    σ, dσdϵ, state = material_response(m, ϵ, state, t_max/num_steps; cache=cache, options=options)
    for k=1:num_steps
        ϵ = SymmetricTensor{2,3}((i,j)-> i==2 && j==1 ? get_ramp_value(m, ϵ21_init, ϵ21_max, k, num_steps) : zero(typeof(ϵ21_max)))
        σ, dσdϵ, state = material_response(m, ϵ, state, t_max/num_steps; cache=cache, options=options)
    end
    return σ, dσdϵ, state
end

@testset "VonMisesPlasticity" begin
    # Basic setup with Voce isotropic hardening and one back-stress of Armstrong-Frederick time

    # constructor
    E=210.e3; ν=0.3; σ_y0=100.0
    m = VonMisesPlasticity(elastic=LinearElastic(E=E, ν=ν),
                           σ_y0=σ_y0,
                           isotropic=(Voce(Hiso=100000.0, κ∞=100.0),), # Can add more dragstresses by more entries in Tuple)
                           kinematic=(ArmstrongFrederick(Hkin=1000000.0, β∞=200.0),)   # Can add more backstresses by more entries in Tuple
    )
    t_max = 1.0
    num_steps = 100
    ϵ11_max = 2.0 * σ_y0 / E
    σ, dσdϵ, state = uniaxial_loading(m, 0.0, ϵ11_max, num_steps, t_max)
    # Check that yield criterion is zero
    σ_full = SymmetricTensor{2,3}((i,j)->i==j && i==1 ? σ[1] : 0.0)
    @test MaterialModels.vonmises(σ_full-sum(state.β)) ≈ σ_y0 + sum(MaterialModels.get_hardening.(m.isotropic, state.λ))
    
    G = E/(2*(1+ν))
    ϵ21_max = 2.0 * (σ_y0/sqrt(3.0))/(2*G)
    σ, dσdϵ, state = shear_loading(m, 0.0, ϵ21_max, num_steps, t_max)
    # Check that yield criterion is zero
    @test MaterialModels.vonmises(σ-sum(state.β)) ≈ σ_y0 + sum(MaterialModels.get_hardening.(m.isotropic, state.λ))

    # Check that error is thrown if material doesn't converge 
    state = initial_material_state(m)
    cache = get_cache(m)
    options = Dict{Symbol, Any}(:nlsolve_params=> Dict{Symbol,Any}(:method=>:newton, :ftol=>1.e-999))
    ϵ = SymmetricTensor{2,3}((i,j)->i==j && i==1 ? ϵ11_max : zero(typeof(ϵ11_max)))
    @test_throws ErrorException σ, dσdϵ, state = material_response(m, ϵ, state, nothing; cache=cache, options=options)
    
end


@testset "VonMisesPlasticity vs Plastic" begin
    # Setup materials
    E = 200.e3; ν=0.3; σ_y0=200.0; Hiso=25.0; κ∞=13.0; Hkin=25.0; β∞=13.0
    plastic = Plastic(E=E, ν=ν, σ_y=σ_y0, H=Hiso+Hkin, r=Hiso/(Hiso+Hkin), κ_∞=κ∞, α_∞=β∞)
    vonmises_plasticity = VonMisesPlasticity(elastic=LinearElastic(E=E, ν=ν), σ_y0=σ_y0,
                                             isotropic=(Voce(Hiso=Hiso, κ∞=κ∞),),
                                             kinematic=(ArmstrongFrederick(Hkin=Hkin, β∞=β∞),))
    # Uniaxial loading test
    t_max = 1.0
    options = Dict{Symbol, Any}(:nlsolve_params=> Dict{Symbol,Any}(:method=>:newton, :ftol=>1.e-12))
    num_steps = 100
    ϵ11_yld = σ_y0 / E
    ϵ11_init = 0.99 * ϵ11_yld
    ϵ11_max = 1.5 * ϵ11_yld
    σ_plastic, dσdϵ_plastic, _ = uniaxial_loading(plastic, ϵ11_init, ϵ11_max, num_steps, t_max, options)
    σ_vmplast, dσdϵ_vmplast, _ = uniaxial_loading(vonmises_plasticity, ϵ11_init, ϵ11_max, num_steps, t_max, options)

    @test σ_plastic ≈ σ_vmplast
    @test isapprox(dσdϵ_plastic, dσdϵ_vmplast; rtol=1.e-4)  # Requires many time steps and low tolerance to avoid small error due to different implementations with default tolerance

    # Shear loading test
    G = E/(2*(1+ν))
    ϵ21_yld = (σ_y0/sqrt(3.0))/(2*G)
    ϵ21_init = 0.9 * ϵ21_yld
    ϵ21_max = 2.0 * ϵ21_yld
    σ_plastic, dσdϵ_plastic, _ = shear_loading(plastic, ϵ21_init, ϵ21_max, num_steps, t_max)
    σ_vmplast, dσdϵ_vmplast, _ = shear_loading(vonmises_plasticity, ϵ21_init, ϵ21_max, num_steps, t_max)

    @test σ_plastic ≈ σ_vmplast
    @test dσdϵ_plastic ≈ dσdϵ_vmplast
end

@testset "VonMisesPlasticity jld2" begin
    E = 200.e3; ν=0.3; σ_y0=200.0; Hiso=25.0; κ∞=13.0; Hkin=25.0; β∞=13.0
    vonmises_plasticity = VonMisesPlasticity(elastic=LinearElastic(E=E, ν=ν), σ_y0=σ_y0,
                                             isotropic=(Voce(Hiso=Hiso, κ∞=κ∞),),
                                             kinematic=(ArmstrongFrederick(Hkin=Hkin, β∞=β∞),))

    loading = get_Plastic_loading()
    check_jld2(vonmises_plasticity, loading, "VonMisesPlasticity1"; OVERWRITE_JLD2=false)
end

get_evolution(af, db, ow, ν, β) = MaterialModels.get_evolution.((af, db, ow), ntuple(i->ν, 3), ntuple(i->β, 3))

@testset "KinematicHardening" begin
    Hkin=10.e3; β∞=30.0
    af = ArmstrongFrederick(Hkin=Hkin, β∞=β∞)
    db = Delobelle(Hkin=Hkin, β∞=β∞, δ=0.5)
    ow = OhnoWang(Hkin=Hkin, β∞=β∞, m=3.0)
    σ_red_dev = dev(rand(SymmetricTensor{2,3}))
    ν = (3/2)*σ_red_dev/MaterialModels.vonmises(σ_red_dev)
    β = zero(SymmetricTensor{2,3})
    
    # Check that all give the same initial hardening modulus
    af_h0, db_h0, ow_h0 = get_evolution(af, db, ow, ν, β)
    @test af_h0 ≈ db_h0
    @test af_h0 ≈ ow_h0

    # Check that ArmstrongFrederick and Delobelle give the same result when β and ν are aligned
    β = (1.0 + rand())*ν    # Ensure scaling > 0
    af_h1, db_h1, ow_h1 = get_evolution(af, db, ow, ν, β)
    @test af_h1 ≈ db_h1     # Should be equal
    @test !(af_h0 ≈ ow_h1)  # Should not be equal
end

@testset "IsotropicHardening" begin
    λ = 0.01 + rand()
    dλ = 1.e-8

    # Voce hardening
    Hiso = 200.0; κ∞=10.0
    voce = Voce(Hiso=Hiso, κ∞=κ∞)
    κ = MaterialModels.get_hardening(voce, λ)
    dκdλ = ForwardDiff.derivative(λarg->MaterialModels.get_hardening(voce, λarg), λ)
    @test dκdλ ≈ Hiso*(1 - κ/κ∞)
    
    # Swift hardening
    K=10.0
    λ0=1.e-3
    n=2.0
    swift = Swift(K=K, λ0=λ0, n=n)
    κ = MaterialModels.get_hardening(swift, λ)
    @test κ ≈ K*(λ0 + λ)^n  # Same equation, but at least protects against changes due to optimizations

end