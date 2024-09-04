abstract type AbstractDim{dim} end

struct Dim{dim} <: AbstractDim{dim} end # generic
struct UniaxialStrain{dim} <: AbstractDim{dim} end # 1D uniaxial strain state
struct UniaxialStress{dim} <: AbstractDim{dim} end # 1D uniaxial stress state
struct PlaneStrain{dim} <: AbstractDim{dim} end # 2D plane strain state
struct PlaneStress{dim} <: AbstractDim{dim} end # 2D plane stress state
struct ShellStress{dim} <: MaterialModels.AbstractDim{dim} end #σ33 == 0.0

# constructors without dim parameter
UniaxialStrain() = UniaxialStrain{1}()
UniaxialStress() = UniaxialStress{1}()
PlaneStrain() = PlaneStrain{2}()
PlaneStress() = PlaneStress{2}()
ShellStress() = ShellStress{3}()

reduce_dim(A::Tensor{1,3}, ::AbstractDim{dim}) where dim = Tensor{1,dim}(i->A[i])
reduce_dim(A::Tensor{2,3}, ::AbstractDim{dim}) where dim = Tensor{2,dim}((i,j)->A[i,j])
reduce_dim(A::Tensor{4,3}, ::AbstractDim{dim}) where dim = Tensor{4,dim}((i,j,k,l)->A[i,j,k,l])
reduce_dim(A::SymmetricTensor{2,3}, ::AbstractDim{dim}) where dim = SymmetricTensor{2,dim}((i,j)->A[i,j])
reduce_dim(A::SymmetricTensor{4,3}, ::AbstractDim{dim}) where dim = SymmetricTensor{4,dim}((i,j,k,l)->A[i,j,k,l])
reduce_dim(A::SymmetricTensor{4,3}, ::AbstractDim{3}) = A

increase_dim(A::Tensor{1,dim,T}) where {dim, T} = Tensor{1,3}(i->(i <= dim ? A[i] : zero(T)))
increase_dim(A::Tensor{2,dim,T}) where {dim, T} = Tensor{2,3}((i,j)->(i <= dim && j <= dim ? A[i,j] : zero(T)))
increase_dim(A::SymmetricTensor{2,dim,T}) where {dim, T} = SymmetricTensor{2,3}((i,j)->(i <= dim && j <= dim ? A[i,j] : zero(T)))
increase_dim(A::SymmetricTensor{2,3}) = A

# restricted strain states
function material_response(
    dim::Union{Dim{d}, UniaxialStrain{d}, PlaneStrain{d}},
    m::AbstractMaterial,
    Δε::AbstractTensor{2,d,T},
    state::AbstractMaterialState,
    Δt = nothing;
    cache = get_cache(m),
    options = Dict{Symbol, Any}(),
    ) where {d,T}

    Δε_3D = increase_dim(Δε)
    σ, dσdε, material_state = material_response(m, Δε_3D, state, Δt; cache=cache, options=options)

    return reduce_dim(σ, dim), reduce_dim(dσdε, dim), material_state
end

# restricted stress states
function material_response(
    dim::Union{UniaxialStress{d}, PlaneStress{d}, ShellStress{d}},
    m::AbstractMaterial,
    Δε::AbstractTensor{2,d,T},
    state::AbstractMaterialState,
    Δt = nothing;
    cache =  get_cache(m),
    options = Dict{Symbol, Any}(),
    ) where {d, T}
    
    tol = get(options, :plane_stress_tol, 1e-8)
    max_iter = get(options, :plane_stress_max_iter, 10)

    Δε_3D = increase_dim(Δε)
    
    zero_idxs = get_zero_indices(dim, Δε_3D)
    nonzero_idxs = get_nonzero_indices(dim, Δε_3D)
    
    for _ in 1:max_iter
        σ, ∂σ∂ε, temp_state = material_response(m, Δε_3D, state, Δt; cache=cache, options=options)
        σ_mandel = _tomandel_sarray(dim, σ)
        if norm(σ_mandel) < tol
            #TODO: Do this with static arrays to avoid transforming to mandel
            _∂σ∂ε = tomandel(SArray, ∂σ∂ε)
            _∂σ∂ε_mod = _∂σ∂ε[nonzero_idxs, nonzero_idxs] - _∂σ∂ε[nonzero_idxs, zero_idxs] * inv(_∂σ∂ε[zero_idxs, zero_idxs]) * _∂σ∂ε[zero_idxs,nonzero_idxs]
            ∂σ∂ε_2D = _frommandel_sarray(dim, _∂σ∂ε_mod) 
            return reduce_dim(σ, dim), ∂σ∂ε_2D, temp_state
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

function _frommandel_sarray(::PlaneStress, A::SMatrix{3,3,T}) where T 
    return frommandel(SymmetricTensor{4,2,T}, A)
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

function _frommandel_sarray(::UniaxialStress, A::SMatrix{1,1,T}) where T 
    return frommandel(SymmetricTensor{4,1,T}, A)
end

# #
# ShellStress
# # 
function _tomandel_sarray(::ShellStress, A::SymmetricTensor{2, 3}) 
    @SVector [A[3,3],]
end

function _tomandel_sarray(::ShellStress, A::SymmetricTensor{4, 3}) 
    @SMatrix [A[3,3,3,3],]
end

function _frommandel_sarray(::ShellStress, A::SVector{1,T}) where T 
    return SymmetricTensor{2,3,T,6}( (0.0, 0.0, 0.0, 0.0, 0.0, A[1]) )
end

function _frommandel_sarray(::ShellStress, v::SMatrix{5,5,T}) where T 
    #v is the stiffness tensor in voight format with stiffnesses associated with the third column and row == 0.0
    #Convert it to tensor notation:
    order = @SMatrix[1 5 4; 0 2 3; 0 0 0]
    return SymmetricTensor{4,3,T}(function (i, j, k, l)
            (i==3 && j==3) && return zero(T)
            (k==3 && l==3) && return zero(T)
            i > j && ((i, j) = (j, i))
            k > l && ((k, l) = (l, k))
            i == j && k == l ? (return v[order[i, j], order[k, l]]) :
            i == j || k == l ? (return T(v[order[i, j], order[k, l]] / √2)) :
                               (return T(v[order[i, j], order[k, l]] / (√2 * √2)))
        end)
end

# out of plane / axis components for restricted stress cases
get_zero_indices(::PlaneStress{2}, ::SymmetricTensor{2,3}) = @SVector[3, 4, 5] # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::PlaneStress{2}, ::SymmetricTensor{2,3}) = @SVector[1, 2, 6] # for voigt/mandel format, do not use on tensor data!
get_zero_indices(::PlaneStress{2}, ::Tensor{2,3}) = @SVector[3, 4, 5, 7, 8] # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::PlaneStress{2}, ::Tensor{2,3}) = @SVector[1, 2, 6, 9] # for voigt/mandel format, do not use on tensor data!

get_zero_indices(::UniaxialStress{1}, ::SymmetricTensor{2,3}) = @SVector[2,3,4,5,6] # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::UniaxialStress{1}, ::SymmetricTensor{2,3}) = @SVector[1] # for voigt/mandel format, do not use on tensor data!
get_zero_indices(::UniaxialStress{1}, ::Tensor{2,3}) = @SVector[2,3,4,5,6,7,8,9] # for voigt/mandel format, do not use on tensor data!
get_nonzero_indices(::UniaxialStress{1}, ::Tensor{2,3}) = @SVector[1] # for voigt/mandel format, do not use on tensor data!

get_zero_indices(::ShellStress{3}, ::SymmetricTensor{2,3}) = @SVector [3,]
get_nonzero_indices(::ShellStress{3}, ::SymmetricTensor{2,3}) = @SVector [1, 2, 4, 5, 6]
