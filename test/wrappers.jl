# constructor
m = LinearElastic(E=200e3, ν=0.3)

# initial state
state = initial_material_state(m)

cache = MaterialModels.PlaneStressCache(zeros(6), zeros(6,6), zeros(6))

dim = MaterialModels.PlaneStress{2}()

Δε = rand(SymmetricTensor{2,2})
σ, ∂σ∂ε, temp_state = material_response(dim, m, Δε, state)
σ, ∂σ∂ε, temp_state = material_response(dim, m, Δε, state; cache = cache);

@test tovoigt(∂σ∂ε) ≈ m.E/(1-m.ν^2)*[1 m.ν 0; m.ν 1 0; 0 0 (1-m.ν)/2]

# timing command
@btime material_response($dim, $m, $Δε, $state, $nothing, options=$(Dict{Symbol, Any}()))
@btime material_response($dim, $m, $Δε, $state, $nothing, cache=$cache, options=$(Dict{Symbol, Any}()))

# 1D case
dim = MaterialModels.UniaxialStress{1}()
Δε = rand(SymmetricTensor{2,1})
σ, ∂σ∂ε, temp_state = material_response(dim, m, Δε, state)
@test ∂σ∂ε[1] ≈ m.E
@test σ ≈ m.E*Δε

@btime material_response($dim, $m, $Δε, $state, $nothing, cache=$cache, options=$(Dict{Symbol, Any}()))