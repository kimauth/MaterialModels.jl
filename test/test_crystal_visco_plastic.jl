

@testset "Plastic" begin
    slipsystems = MaterialModels.slipsystems(MaterialModels.FCC(), rand(RodriguesParam))
    m = MaterialModels.CrystalViscoPlastic(E=200e3, ν=0.3, τ_y=400., H_iso=1e3, H_kin=1e3, q=0.0, α_∞=100., t_star=20., σ_c=50., m=10., slipsystems=slipsystems)
end