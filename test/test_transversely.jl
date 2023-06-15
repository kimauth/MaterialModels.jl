# constructor

@testset "TransverselyIsotropic" begin
    E_L = 100e3
    E_T = 10e3
    G_LT = 5e3
    ν_LT = 0.4 
    ν_TT = 0.3
    m = TransverselyIsotropic(; E_L, E_T, G_LT, ν_LT, ν_TT)

    # initial state
    test_cache = get_cache(m)
    @test isnothing(test_cache)
    
    ε = rand(SymmetricTensor{2,3})
    # material_response
    state = initial_material_state(m, Vec((1.0, 0.0, 0.0)))
    σ, ∂σ∂ε, temp_state = material_response(m, ε, state)
    @test σ ≈ ∂σ∂ε ⊡ ε

    #Rotational symmetry
    state180 = initial_material_state(m, Vec((-1.0, 0.0, 0.0)))
    _, ∂σ∂ε180, _ = material_response(m, ε, state180)
    @test ∂σ∂ε ≈ ∂σ∂ε180

    #Rotatating the strain around 1-axis should not effect anything
    base1 = basevec(Vec{3}) 
    base2 = let α = deg2rad(45)
        (Vec((1.0,0.0,0.0)), Vec((0.0,cos(α),sin(α))), Vec((0.0,-sin(α),cos(α))))
    end
    R = Tensor{2,3}( (i,j) -> base1[i] ⋅ base2[j] )
    ε45 = symmetric(R' ⋅ ε ⋅ R)
    _, ∂σ∂ε45, _ = material_response(m, ε45, state)
    @test ∂σ∂ε45 ≈ ∂σ∂ε

    #Compare with engineering way of implementing transversely isotropic materials
    # https://en.wikipedia.org/wiki/Transverse_isotropy
    G_TT = E_T/2(1+ν_TT)
    _C = [   1/E_L -ν_LT/E_L -ν_LT/E_L   0       0         0;
         -ν_LT/E_L     1/E_T -ν_TT/E_T   0       0         0;
         -ν_LT/E_L -ν_TT/E_T     1/E_T   0       0         0;
             0         0        0        1/G_TT  0         0;
             0         0        0        0       1/G_LT    0;
             0         0        0        0       0         1/G_LT]
    
    C = tovoigt(inv(∂σ∂ε), offdiagscale=2.0)
    @test _C ≈ C
    #=components = ((1,1),(2,2),(3,3),(2,3),(1,3),(1,2))
    for (I,(i,j)) in pairs(components), (J,(k,l)) in pairs(components)
        if !( _C[I,J] ≈ C[i,j,k,l] )
            @show _C[I,J]  C[i,j,k,l]
            @show (I,J) (i,j,k,l)
        end
    end=#
end

function get_TransverselyIsotropic_loading()
    loading = range(0.0,  0.02, length=2)
    return [SymmetricTensor{2,3}((ε, ε/10, 0.0, 0.0, 0.0, 0.0)) for ε in loading]
end  

@testset "TransverselyIsotropic jld2" begin
    E_L = 100e3, 
    E_T = 10e3
    G_LT = 5e3
    ν_LT = 0.4 
    ν_TT = 0.3
    m = TransverselyIsotropicEngineeringConstants(; E_L, E_T, G_LT, ν_LT, ν_TT)
    loading = get_TransverselyIsotropic_loading()
    check_jld2(m, loading, "TransverselyIsotropic")#, OVERWRITE_JLD2=false)
end