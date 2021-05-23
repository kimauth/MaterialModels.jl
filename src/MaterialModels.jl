module MaterialModels

using Reexport
@reexport using Tensors
using NLsolve
using Rotations

# Write your package code here.
abstract type AbstractMaterial end
abstract type AbstractMaterialState end
abstract type AbstractResiduals end


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

include("LinearElastic.jl")
include("Plastic.jl")
include("CrystalViscoPlastic/slipsystems.jl")
include("CrystalViscoPlastic/CrystalViscoPlastic.jl")

include("nonlinear_solver.jl")

export initial_material_state, get_cache
export constitutive_driver

export LinearElastic, Plastic
export LinearElasticState, PlasticState

end
