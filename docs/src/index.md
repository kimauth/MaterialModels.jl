# Materials
## Elastic materials
```@docs
LinearElastic
material_response(m::LinearElastic, ε::SymmetricTensor{2,3}, state::LinearElasticState)
```
## Plastic materials
```@docs
Plastic
material_response(m::Plastic, Δε::SymmetricTensor{2,3,T,6}, state::PlasticState{3}; kwargs...) where T
VonMisesPlasticity
material_response(m::VonMisesPlasticity, ϵ::SymmetricTensor{2,3}, old::MaterialModels.VonMisesPlasticityState; kwargs...)
```
### Isotropic hardening laws
```@docs
Voce
Swift
```
### Kinematic hardening laws
```@docs
ArmstrongFrederick
Delobelle
OhnoWang
```