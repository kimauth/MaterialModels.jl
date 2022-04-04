using MaterialModels
using Test
using Rotations
using StaticArrays
using JLD2

include("test_utils.jl")
include("test_newton_solver.jl")
include("test_linear_elastic.jl")
include("test_plastic.jl")
include("test_crystal_visco_plastic.jl")
# include("test_crystal_visco_plastic_red.jl")
include("test_wrappers.jl")
include("test_neohook.jl")
include("test_yeoh.jl")
include("test_stvenant.jl")
include("test_straintraits.jl")
include("test_xu_needleman.jl")