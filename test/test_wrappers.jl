@testset "wrappers" begin
    m = LinearElastic(E=200e3, ν=0.3)
    state = initial_material_state(m)

    ################################################
    # stress wrapper
    ################################################
    # 1D
    dim = UniaxialStress()
    Δε = rand(SymmetricTensor{2,1})
    σ, ∂σ∂ε, temp_state = material_response(dim, m, Δε, state)
    @test ∂σ∂ε[1] ≈ m.E
    @test σ ≈ m.E*Δε

    # 2D
    dim = PlaneStress()
    Δε = rand(SymmetricTensor{2,2})
    σ, ∂σ∂ε, temp_state = material_response(dim, m, Δε, state)
    @test σ ≈ ∂σ∂ε ⊡ Δε
    @test tovoigt(∂σ∂ε) ≈ m.E/(1-m.ν^2)*[1 m.ν 0; m.ν 1 0; 0 0 (1-m.ν)/2]

    # Shell stress σ33 == 0.0
    dim = ShellStress()
    Δε = rand(SymmetricTensor{2,3})
    σ, ∂σ∂ε, temp_state = material_response(dim, m, Δε, state)
    ν = m.ν
    a = (1-ν)/2
    ∂σ∂ε_shell = m.E/(1-m.ν^2) * [1 ν 0 0 0 0; #From "The Finite Element Method - Linear static and dynamic FEM" by Thomas J.R Hughes
                                  ν 1 0 0 0 0;
                                  0 0 0 0 0 0
                                  0 0 0 a 0 0;
                                  0 0 0 0 a 0;
                                  0 0 0 0 0 a]
    @test σ[3,3] ≈ 0.0 atol=1e-10
    @test σ ≈ ∂σ∂ε ⊡ Δε
    @test tovoigt(∂σ∂ε) ≈ ∂σ∂ε_shell
    
    ################################################
    # strain wrapper
    ################################################
    # 1D
    dim = UniaxialStrain()
    Δε = rand(SymmetricTensor{2,1})
    σ, ∂σ∂ε, temp_state = material_response(dim, m, Δε, state)
    Δε_3D = SymmetricTensor{2,3}((i,j)-> i==1 && j==1 ? Δε[1,1] : 0.0)
    σ_3D, ∂σ∂ε_3D, temp_state = material_response(m, Δε_3D, state)
    @test σ[1,1] == σ_3D[1,1]
    @test ∂σ∂ε[1,1,1,1] == ∂σ∂ε_3D[1,1,1,1]

    # 2D
    dim = PlaneStrain()
    Δε = rand(SymmetricTensor{2,2})
    σ, ∂σ∂ε, temp_state = material_response(dim, m, Δε, state)
    @test σ ≈ ∂σ∂ε ⊡ Δε
    @test tovoigt(∂σ∂ε) ≈ m.E/(1+m.ν)/(1-2m.ν)*[1-m.ν m.ν 0; m.ν 1-m.ν 0; 0 0 (1-2m.ν)/2]

end

