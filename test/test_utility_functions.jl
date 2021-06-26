@testset "utility functions" begin
    s = rand() + 1.0
    σ = SymmetricTensor{2,3}((i,j)-> i==j && i==1 ? s : 0.0)
    @test MaterialModels.vonmises(σ) ≈ s   # Uniaxial stress should give the correct value

    σ = SymmetricTensor{2,3}((i,j)-> i==2 && j==1 ? s : 0.0)
    @test MaterialModels.vonmises(σ) ≈ √3*s    # Shear stress have factor √3

    σ = rand(SymmetricTensor{2,3})
    @test MaterialModels.vonmises(σ) ≈ MaterialModels.vonmises_dev(dev(σ))    # Check that the deviatoric version works as intended

    x = 1.0 + rand()
    @test MaterialModels.macaulay(-x) ≈ zero(x)
    @test MaterialModels.macaulay(x) ≈ x

end