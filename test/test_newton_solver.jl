@testset "Newton solver" begin
    f1(x) = 2x + 1
    solver_result = solve(f1, 0.0)
    @test !isnothing(solver_result)
    x, dfdx = solver_result
    @test x == -0.5
    @test dfdx == 2.0

    f2(x) = exp(x) - 1.0
    solver_result = solve(f2, 10.)
    @test !isnothing(solver_result)
    x, dfdx = solver_result
    @test f2(x) < 1e-8
    @test dfdx ≈ exp(x)
end
