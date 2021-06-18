module MaterialModels

using Reexport
@reexport using Tensors
using NLsolve
using Rotations
import ForwardDiff
import DiffResults
using StaticArrays


"""
    AbstractMaterial

Store material parameters here. It can also be used to store constant precomputed properties, e.g. the elastic stiffness tensor.
Ideally, the name should be chosen according to the first author of the initial publication of a model. For every `Material` there should be a keyword constructor
and a list of arguments in the docstrings. If possible, also include the reference to a publication.
"""
abstract type AbstractMaterial end

"""
    AbstractMaterialState

Store state variables here. For now, this should **not** be mutable, a new object should be constructed for every new state. (We can discuss if this is a good or a bad idea.)
"""
abstract type AbstractMaterialState end
abstract type AbstractResiduals end

"""
    material_response(m::AbstractMaterial, Δε::SymmetricTensor{2,3}, state::AbstractMaterialState, Δt; cache, options)

Compute the stress, stress tangent and state variables for the given strain increment `Δε` and previous state `state`.

Instead of the strain increment, the total strain could be handed over. Good ideas on how to handle this in general are welcome.
For non-continuum kind of material models, the interface should be similar with stress-like and strain-like quantities.
(E.g. for cohesive laws traction instead of stress and displacement jump instead of strain.)
This function signature must be the same for all material models, even if they don't require all arguments.

"""
function material_response end

"""
    initial_material_state(::AbstractMaterial)

Return the `MaterialState` that belongs to the given `Material` and is initialized with zeros. 
"""
function initial_material_state end

"""
    get_cache(m::AbstractMaterial)

Construct cache object for iteratively solving non-linear material models.

For material models which require an iterative solution procedure, it is recommended to allocate storage for the iterative solver only once and reuse it for all material points.
When multithreading is used, each threads needs its own cache.
"""
function get_cache end
"""
    update_cache!(cache::OnceDifferentiable, f)

Update the cache object with the residual function for the current time/load step.

As the residual functions depend i.a. on the strain increment, the function and its jacobian need to be updated for every load step.
"""
function update_cache! end


include("LinearElastic.jl")
include("Plastic.jl")
include("CrystalViscoPlastic/slipsystems.jl")
include("CrystalViscoPlastic/CrystalViscoPlastic.jl")
include("CrystalViscoPlastic/CrystalViscoPlasticRed.jl")
include("SmallStrainPlasticity/SmallStrainPlasticity.jl")

include("nonlinear_solver.jl")
include("wrappers.jl")

export initial_material_state, get_cache
export material_response

export AbstractMaterial
export LinearElastic, Plastic
export LinearElasticState, PlasticState
export OneD, UniaxialStrain, UniaxialStress, PlaneStrain, PlaneStress

export Chaboche
export LinearIsotropicElasticity
export Voce, Swift
export ArmstrongFrederick, Delobelle, OhnoWang

end
