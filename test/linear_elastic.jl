# constructor
m = LinearElastic(E=200e3, ν=0.3)

# initial state
state = initial_material_state(m)
@test state.σ == zero(SymmetricTensor{2,3})

# constitutive driver
Δε = rand(SymmetricTensor{2,3})
σ, ∂σ∂ε, temp_state = constitutive_driver(m, Δε, state)
@test σ == temp_state.σ
@test ∂σ∂ε == m.Eᵉ
