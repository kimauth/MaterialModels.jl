@testset "StVenant" begin
    m = StVenant(μ=1.0, λ=1.0)

    F = rand(Tensor{2,3})
    C = tdot(F)
    E = (C - one(C))/2

    S, dSdC = material_response(m, C)
    d²ΨdC², dΨdC = hessian(C->elastic_strain_energy_density(m, C), C, :all)
    @test S ≈ 2dΨdC
    @test dSdC ≈ 2d²ΨdC²

    d²ΨdE², dΨdE = hessian(E->elastic_strain_energy_density(m, 2E + one(E)), E, :all)
    @test 2dΨdC ≈ dΨdE
    @test 4d²ΨdC²≈ d²ΨdE²

    dSdC_autodiff = gradient(C->material_response(m, C)[1], C)
    @test dSdC_autodiff ≈ dSdC
end
