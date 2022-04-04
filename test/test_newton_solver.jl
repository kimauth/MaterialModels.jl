@testset "Newton solver" begin
    f(x) = 2x + 1
    solver_result = solve(f, 0.0)
    @test !isnothing(solver_result)
    x, dfdx = solver_result
    @test x == -0.5
    @test dfdx == 2.0

    f(x) = exp(x) - 1.0
    solver_result = solve(f, 10.)
    @test !isnothing(solver_result)
    x, dfdx = solver_result
    @test f(x) < 1e-8
    @test dfdx ≈ exp(x)
end