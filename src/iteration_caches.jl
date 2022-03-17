"""
AbstractCache
Stores matrices, vectors etc. to avoid re-allcating memory each time the material routine is called.
"""
abstract type AbstractCache end


# wraps OnceDiffentiable, can't make this a subtype of AbstractCache without wrapping
struct SolverCache{C} <: AbstractCache
    cache::C
end

struct CacheContainer{C1, C2} <: AbstractCache
    solver_cache::C1
    plane_stress_cache::C2
end

# fallback in case there is no cache defined
struct PlaneStressCache{TF, TDF, TX}
    F::TF
    DF::TDF
    x_f::TX
end

# generic fallback, Materials without field σ need to define it
get_stress_type(state::AbstractMaterialState) = typeof(state.σ)

# fallback for optional cache argument
function get_plane_stress_cache(m::AbstractMaterial)
    state = initial_material_state(m)
    stress_type = get_stress_type(state)
    T = eltype(stress_type)
    M = Tensors.n_components(Tensors.get_base(stress_type))
    cache = PlaneStressCache(Vector{T}(undef,M), Matrix{T}(undef,M,M), Vector{T}(undef,M))
    return cache
end

# cache constructors
_get_cache(::AbstractMaterial) = nothing # fallback, models that require iterative solving have to implement this

function get_cache(m::AbstractMaterial)
    c = _get_cache(m)
    isnothing(c) && @warn("$(typeof(m)) doesn't require a cache.")
    return SolverCache(_get_cache(m))
end

get_cache(m::AbstractMaterial, dim::AbstractDim) = get_cache(m)

function get_cache(m::AbstractMaterial, dim::Union{PlaneStress, UniaxialStress})
    solver_cache = _get_cache(m)
    plane_stress_cache = get_plane_stress_cache(m)
    if isnothing(solver_cache)
        return plane_stress_cache
    else
        return CacheContainer(solver_cache, plane_stress_cache)
    end
end

# accessor functions
solver_cache(c::SolverCache) = c.cache
solver_cache(c::CacheContainer) = c.solver_cache
plane_stress_cache(c::PlaneStressCache) = c
plane_stress_cache(c::CacheContainer) = c.plane_stress_cache
