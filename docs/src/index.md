# Materials
```@docs
LinearElastic
material_response(m::LinearElastic, Δε::SymmetricTensor{2,3}, state::LinearElasticState{3})
Plastic
material_response(m::Plastic, Δε::SymmetricTensor{2,3,T,6}, state::PlasticState{3}; kwargs...) where T
```
