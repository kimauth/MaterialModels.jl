@testset "XuNeedleman" begin
    σₘₐₓ = 400.
    τₘₐₓ=200.
    δₙ=0.002
    δₜ=0.002
    Φₙ = xu_needleman_Φₙ(σₘₐₓ, δₙ)
    Φₜ = xu_needleman_Φₜ(τₘₐₓ, δₜ)
    # test parameter conversion
    @test xu_needleman_σₘₐₓ(Φₙ, δₙ) ≈ σₘₐₓ
    @test xu_needleman_τₘₐₓ(Φₜ, δₜ) ≈ τₘₐₓ
    
    m = XuNeedleman(σₘₐₓ = σₘₐₓ, τₘₐₓ=τₘₐₓ, Φₙ=Φₙ, Φₜ=Φₜ, Δₙˢ=0.01)

    # 3D
    Δ = Tensor{1,3}((0.0, 0.0, δₙ))
    T, dTdΔ, state = material_response(m, Δ)
    @test T[end] ≈ σₘₐₓ
    Δ = Tensor{1,3}((sqrt(2)*δₜ/2, 0.0, 0.0))
    T, dTdΔ, state = material_response(m, Δ)
    @test T[1] ≈ τₘₐₓ

    # 2D
    Δ = Tensor{1,2}((0.0, δₙ))
    T, dTdΔ, state = material_response(m, Δ)
    @test T[end] ≈ σₘₐₓ
    Δ = Tensor{1,2}((sqrt(2)*δₜ/2, 0.0))
    T, dTdΔ, state = material_response(m, Δ)
    @test T[1] ≈ τₘₐₓ
end