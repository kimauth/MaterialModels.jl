abstract type AbstractDim end

struct UniaxialStrain <: AbstractDim end # 1D uniaxial strain state
struct UniaxialStress <: AbstractDim end # 1D uniaxial stress state
struct PlaneStrain    <: AbstractDim end # 2D plane strain state
struct PlaneStress    <: AbstractDim end # 2D plane stress state
struct ThreeD         <: AbstractDim end # 3D

getdim(::Type{UniaxialStrain}) = 1
getdim(::Type{UniaxialStress}) = 1
getdim(::Type{PlaneStrain}) = 2
getdim(::Type{PlaneStress}) = 2
getdim(::Type{ThreeD}) = 3

getdim(d::T) where T<:AbstractDim = getdim(typeof(d))

reduce_dim(A::Tensor{1,3}, d::AbstractDim) = Tensor{1,getdim(d)}(i->A[i])
reduce_dim(A::Tensor{2,3}, d::AbstractDim) = Tensor{2,getdim(d)}((i,j)->A[i,j])
reduce_dim(A::Tensor{4,3}, d::AbstractDim) = Tensor{4,getdim(d)}((i,j,k,l)->A[i,j,k,l])
reduce_dim(A::SymmetricTensor{2,3}, d::AbstractDim) = SymmetricTensor{2,getdim(d)}((i,j)->A[i,j])
reduce_dim(A::SymmetricTensor{4,3}, d::AbstractDim) = SymmetricTensor{4,getdim(d)}((i,j,k,l)->A[i,j,k,l])

increase_dim(A::Tensor{1,dim,T}) where {dim,T} = Tensor{1,3}(i->(i <= dim ? A[i] : zero(T)))
increase_dim(A::Tensor{2,dim,T}) where {dim,T} = Tensor{2,3}((i,j)->(i <= dim && j <= dim ? A[i,j] : zero(T)))
increase_dim(A::SymmetricTensor{2,dim,T}) where {dim,T} = SymmetricTensor{2,3}((i,j)->(i <= dim && j <= dim ? A[i,j] : zero(T)))

# 3d materials pass through
function material_response(
    ::ThreeD,
    m::AbstractMaterial,
    Δε::AbstractTensor{2,3,T},
    state::AbstractMaterialState,
    Δt = nothing;
    cache = nothing,
    options = Dict{Symbol, Any}(),
    ) where {T}

    return material_response(m, Δε, state, Δt; cache=cache, options=options)

end

# restricted strain states
function material_response(
    dimstate::Union{UniaxialStrain, PlaneStrain},
    m::AbstractMaterial,
    Δε::AbstractTensor{2,d,T},
    state::AbstractMaterialState,
    Δt = nothing;
    cache = nothing,
    options = Dict{Symbol, Any}(),
    ) where {d,T}

    @assert(getdim(dimstate) == d)

    Δε_3D = increase_dim(Δε)
    σ, dσdε, material_state = material_response(m, Δε_3D, state, Δt; cache=cache, options=options)

    return reduce_dim(σ, dimstate), reduce_dim(dσdε, dimstate), material_state
end

# restricted stress states
function material_response(
    dimstate::Union{UniaxialStress, PlaneStress},
    m::AbstractMaterial,
    Δε::AbstractTensor{2,d,T},
    state::AbstractMaterialState,
    Δt = nothing;
    cache::Union{Any, Nothing} = nothing, #get_cache(m), #TODO: create AbstractCache type
    options = Dict{Symbol, Any}(),
    ) where {d, T}
    
    @assert(getdim(dimstate) == d)

    tol = get(options, :plane_stress_tol, 1e-8)
    max_iter = get(options, :plane_stress_max_iter, 10)

    Δε_3D = increase_dim(Δε)
    
    #zero_idxs = get_zero_indices(dimstate, Δε_3D)
    nonzero_idxs = get_nonzero_indices(dimstate, Δε_3D)
    
    for _ in 1:max_iter
        σ, ∂σ∂ε, temp_state = material_response(m, Δε_3D, state, Δt; cache=cache, options=options)
        σ_mandel = _tomandel_sarray(dim, σ)
        if norm(σ_mandel) < tol
            ∂σ∂ε_2D = fromvoigt(SymmetricTensor{4,d}, inv(inv(tovoigt(∂σ∂ε))[nonzero_idxs, nonzero_idxs])) #TODO: Maybe solve this with static arrays aswell?
            return reduce_dim(σ, dimstate), ∂σ∂ε_2D, temp_state
        end
        J = _tomandel_sarray(dim, ∂σ∂ε)
        Δε_temp = -inv(J)*σ_mandel
        Δε_correction = _frommandel_sarray(dim, Δε_temp)
        Δε_3D = Δε_3D + Δε_correction
    end
    error("Not converged. Could not find plane stress state.")
end

function _tomandel_sarray(::PlaneStress, A::SymmetricTensor{2, 3}) 
    @SVector [A[3,3], √2A[2,3], √2A[1,3]]
end

function _tomandel_sarray(::PlaneStress, A::SymmetricTensor{4, 3}) 
    @SMatrix [    A[3,3,3,3] √2*A[3,3,2,3] √2*A[3,3,1,3];
               √2*A[2,3,3,3]  2*A[2,3,2,3]  2*A[2,3,1,3] ;
               √2*A[1,3,3,3]  2*A[1,3,2,3]  2*A[1,3,1,3] ]
end

function _frommandel_sarray(::PlaneStress, A::SVector{3,T}) where T 
    return SymmetricTensor{2,3,T,6}( (0.0, 0.0, A[3]/√2, 0.0, A[2]/√2, A[1]) )
end

function _tomandel_sarray(::UniaxialStress, A::SymmetricTensor{2, 3}) 
    @SVector [A[2,2], A[3,3], √2A[2,3], √2A[1,3], √2A[1,2]]
end

function _tomandel_sarray(::UniaxialStress, A::SymmetricTensor{4, 3}) 
    @SMatrix [  A[2,2,2,2] A[2,2,3,3] √2*A[2,2,2,3] √2*A[2,2,1,3] √2*A[2,2,1,2];
                A[3,3,2,2] A[3,3,3,3] √2*A[3,3,2,3] √2*A[3,3,1,3] √2*A[3,3,1,2];
                √2*A[2,3,2,2] √2*A[2,3,3,3] 2*A[2,3,2,3] 2*A[2,3,1,3] 2*A[2,3,1,2];
                √2*A[1,3,2,2] √2*A[1,3,3,3] 2*A[1,3,2,3] 2*A[1,3,1,3] 2*A[1,3,1,2];
                √2*A[1,2,2,2] √2*A[1,2,3,3] 2*A[1,2,2,3] 2*A[1,2,1,3] 2*A[1,2,1,2]]

end

function _frommandel_sarray(::UniaxialStress, A::SVector{5,T}) where T 
    return SymmetricTensor{2,3,T,6}( (0.0, A[5]/√2, A[4]/√2, A[1], A[3]/√2, A[2]) )
end 


# out of plane / axis components for restricted stress cases
get_zero_indices(::PlaneStress{2}, ::SymmetricTensor{2,3}) = [3, 4, 5] # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::PlaneStress{2}, ::SymmetricTensor{2,3}) = [1, 2, 6] # for voigt/mandel format, do not use on tensor data!
get_zero_indices(::PlaneStress{2}, ::Tensor{2,3}) = [3, 4, 5, 7, 8] # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::PlaneStress{2}, ::Tensor{2,3}) = [1, 2, 6, 9] # for voigt/mandel format, do not use on tensor data!

get_zero_indices(::UniaxialStress{1}, ::SymmetricTensor{2,3}) = collect(2:6) # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::UniaxialStress{1}, ::SymmetricTensor{2,3}) = [1] # for voigt/mandel format, do not use on tensor data!
get_zero_indices(::UniaxialStress{1}, ::Tensor{2,3}) = collect(2:9) # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::UniaxialStress{1}, ::Tensor{2,3}) = [1] # for voigt/mandel format, do not use on tensor data!
