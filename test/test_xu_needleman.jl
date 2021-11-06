@testset "XuNeedleman" begin
    m = MaterialModels.XuNeedleman(σₘₐₓ = 400., τₘₐₓ=200., δₙ=0.002, δₜ=0.002, Δₙˢ=0.01)

    # 3D
    Δ = Tensor{1,3}((0.0, 0.0, m.δₙ))
    T, dTdΔ, state = material_response(m, Δ)
    @test T[end] ≈ m.σₘₐₓ
    Δ = Tensor{1,3}((sqrt(2)*m.δₜ/2, 0.0, 0.0))
    T, dTdΔ, state = material_response(m, Δ)
    @test T[1] ≈ m.τₘₐₓ

    # 2D
    Δ = Tensor{1,2}((0.0, m.δₙ))
    T, dTdΔ, state = material_response(m, Δ)
    @test T[end] ≈ m.σₘₐₓ
    Δ = Tensor{1,2}((sqrt(2)*m.δₜ/2, 0.0))
    T, dTdΔ, state = material_response(m, Δ)
    @test T[1] ≈ m.τₘₐₓ
end