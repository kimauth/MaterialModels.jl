# Materials
```@docs
LinearElastic
material_response(m::LinearElastic, ε::SymmetricTensor{2,3}, state::LinearElasticState)
Plastic
material_response(m::Plastic, Δε::SymmetricTensor{2,3,T,6}, state::PlasticState{3}; kwargs...) where T
VonMisesPlasticity
material_response(m::VonMisesPlasticity, ε::SymmetricTensor{2,3}, state, Δt; kwargs...) where T
```
