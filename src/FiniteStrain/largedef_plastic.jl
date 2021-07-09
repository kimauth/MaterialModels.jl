export MatHyperElasticPlastic, MatHyperElasticPlasticState

"""
    MatHyperElasticPlastic <: AbstractMaterial

Hyper elastic plastic material for large deformations
"""
struct MatHyperElasticPlastic{M <: AbstractMaterial} <: AbstractMaterial
    density::Float64
    elastic_material::M #Elastic material, Yeoh or Neo-hook
    τ₀::Float64		    #Yield stress
    H::Float64		    #Hardening
end

struct MatHyperElasticPlasticState <: AbstractMaterialState
    ϵᵖ::Float64					       #Plastic strain
    Fᵖ::Tensor{2,3,Float64,9}		   #Plastic deformation grad
    ν::Tensor{2,3,Float64,9}	
end

struct MatHyperElasticPlasticExtras <: AbstractExtras
    D  ::Float64 #Dissipation
    ∂D ::SymmetricTensor{2,3,Float64,6}
end

# # # # # # #
# Constructors
# # # # # # #

function init_material_state(::MatHyperElasticPlastic)
    Fᵖ = one(Tensor{2,3,Float64})
    ν = zero(Tensor{2,3,Float64})
    return MatHyperElasticPlasticState(0.0, Fᵖ, ν)
end

function MatHyperElasticPlastic(; 
    elastic_material::HyperElasticMaterial,
    τ₀  ::T,
    H   ::T,
    density ::T = NaN, 
    ) where T

    return MatHyperElasticPlastic(density, elastic_material, τ₀, H)
end

# # # # # # #
# Drivers
# # # # # # #

function _compute_2nd_PK(mp::MatHyperElasticPlastic, C::SymmetricTensor{2,dim,T}, state::MatHyperElasticPlasticState, compute_dissipation::Bool) where {dim,T}
    emat, τ₀, H = (mp.elastic_material, mp.τ₀, mp.H)
    
    I = one(C)
    Iᵈᵉᵛ = otimesu(I,I) - 1/3*otimes(I,I)
    
    ϵᵖ = state.ϵᵖ
    Fᵖ = state.Fᵖ
    ν = state.ν
    
    dγ = 0.0
    
    phi, Mᵈᵉᵛ, Cᵉ, S̃, ∂S̃∂Cₑ = hyper_yield_function(C, state.Fᵖ, state.ϵᵖ, emat, τ₀, H)
    dFᵖdC = zero(Tensor{4,dim,T})
    dΔγdC = zero(Tensor{2,dim,T})
    dgdC = zero(Tensor{2,dim,T})
    g = 0.0

    if phi > 0 #Plastic step
        #Setup newton varibles
        TOL = 1e-9
        newton_error, counter = TOL + 1, 0
        dγ = 0.0
        
        local ∂ϕ∂Mᵈᵉᵛ, ∂Mᵈᵉᵛ∂M, ∂M∂Cₑ, dϕdΔγ, ∂ϕ∂M, ∂M∂S̃
        while true; counter +=1 

            Fᵖ = state.Fᵖ - dγ*state.Fᵖ⋅state.ν
            ϵᵖ = state.ϵᵖ + dγ

            R, Mᵈᵉᵛ, Cₑ, S̃, ∂S̃∂Cₑ = _hyper_yield_function(C, Fᵖ, ϵᵖ, emat, τ₀, H) 
            
            #Numerical diff
            #h = 1e-6
            #J = (yield_function(C, state.Fᵖ - (dγ+h)*state.Fᵖ⋅state.ν, state.ϵᵖ + (dγ+h), μ, λ, τ₀, H)[1] - yield_function(C, state.Fᵖ - dγ*state.Fᵖ⋅state.ν, state.ϵᵖ + dγ, μ, λ, τ₀, H)[1]) / h

            ∂ϕ∂M = sqrt(3/2) * (Mᵈᵉᵛ/norm(Mᵈᵉᵛ)) #⊡ Iᵈᵉᵛ #  state.ν#
            ∂M∂Cₑ = otimesu(I, S̃)
            ∂Cₑ∂Fᵖ = otimesl(I, transpose(Fᵖ)⋅C) + otimesu(transpose(Fᵖ)⋅C, I)
            ∂Fᵖ∂Δγ = -state.Fᵖ⋅state.ν
            ∂M∂S̃ = otimesu(Cₑ, I)

            dϕdΔγ = ∂ϕ∂M ⊡ ((∂M∂Cₑ + ∂M∂S̃ ⊡ ∂S̃∂Cₑ) ⊡ ∂Cₑ∂Fᵖ ⊡ ∂Fᵖ∂Δγ) - H
             
            ddγ = dϕdΔγ\-R
            dγ += ddγ

            newton_error = norm(R)

            if newton_error < TOL
                break
            end

            if counter > 6
                #@warn("Could not find equilibrium in material")
                break
            end
        end
        
        ∂Cₑ∂C = otimesu(transpose(Fᵖ),transpose(Fᵖ))
        ∂ϕ∂C = ∂ϕ∂M ⊡ ((∂M∂Cₑ + ∂M∂S̃ ⊡ ∂S̃∂Cₑ) ⊡ ∂Cₑ∂C) 
        dΔγdC = -inv(dϕdΔγ) * ∂ϕ∂C
        dFᵖdC = -(state.Fᵖ⋅state.ν) ⊗ dΔγdC

    end

    
    ν = norm(Mᵈᵉᵛ)==0.0 ? zero(Mᵈᵉᵛ) : sqrt(3/2)*Mᵈᵉᵛ/(norm(Mᵈᵉᵛ))

    S = (Fᵖ ⋅ S̃ ⋅ transpose(Fᵖ))

    ∂S∂Fᵖ = otimesu(I, (Fᵖ ⋅ S̃)) + otimesl((Fᵖ ⋅ S̃), I)
    ∂S∂S̃ = otimesu(Fᵖ,Fᵖ)
    ∂Cₑ∂Fᵖ = otimesl(I, transpose(Fᵖ) ⋅ C) + otimesu(transpose(Fᵖ) ⋅ C, I)
    ∂Cₑ∂C = otimesu(transpose(Fᵖ),transpose(Fᵖ))
    dCₑdC = ∂Cₑ∂C + ∂Cₑ∂Fᵖ ⊡ dFᵖdC
    dSdC = ∂S∂Fᵖ ⊡ dFᵖdC + ∂S∂S̃ ⊡ ∂S̃∂Cₑ ⊡ dCₑdC

    g = 0.0, 
    dgdC = zero(SymmetricTensor{2,3,T,6})
    if compute_dissipation
        if phi > 0.0
            M = Cᵉ ⋅ S̃
            g, dgdC = _compute_dissipation(M, ν, dγ, dCₑdC, ∂S̃∂Cₑ, dΔγdC, ∂M∂Cₑ, ∂M∂S̃, Iᵈᵉᵛ, I)
        end
    end

    return symmetric(S), dSdC, ϵᵖ, ν, Fᵖ, g, dgdC
end

function _hyper_yield_function(C::SymmetricTensor, Fᵖ::Tensor, ϵᵖ::T, emat::MatNeoHook, τ₀::Float64, H::Float64) where T
    Cᵉ = symmetric(transpose(Fᵖ)⋅C⋅Fᵖ)

    S̃, ∂S̃∂Cₑ = _constitutive_driver(emat, Cᵉ)

    #Mandel stress
    Mbar = Cᵉ⋅S̃
    Mᵈᵉᵛ = dev(Mbar)

    yield_func = sqrt(3/2)*norm(Mᵈᵉᵛ) - (τ₀ + H*ϵᵖ)

    return yield_func, Mᵈᵉᵛ, Cᵉ, S̃, ∂S̃∂Cₑ
end

function _compute_dissipation(M, ν, Δγ, dCₑdC, ∂S̃∂Cₑ, dΔγdC, ∂M∂Cₑ, ∂M∂S̃, Iᵈᵉᵛ, I)

    Mᵈᵉᵛ = dev(M)

    ∂g∂M = ν * Δγ
    ∂g∂ν = M * Δγ
    ∂g∂Δγ = M ⊡ ν
    
    ∂ν∂Mᵈᵉᵛ = √(3/2) * (1/norm(Mᵈᵉᵛ)) * (Iᵈᵉᵛ - (Mᵈᵉᵛ ⊗ Mᵈᵉᵛ)/(norm(Mᵈᵉᵛ)^2) )
    ∂Mᵈᵉᵛ∂M = Iᵈᵉᵛ
    dMdC = (∂M∂Cₑ + ∂M∂S̃ ⊡ ∂S̃∂Cₑ) ⊡ dCₑdC

    g = M ⊡ ν*Δγ
    dgdC = (∂g∂M + ∂g∂ν ⊡ ∂ν∂Mᵈᵉᵛ ⊡ ∂Mᵈᵉᵛ∂M) ⊡ dMdC  +  ∂g∂Δγ * dΔγdC 

    return g, dgdC
end

function material_response(mp::MatHyperElasticPlastic, C::SymmetricTensor{2,3,T,6}, state::MatHyperElasticPlasticState, Δt=0.0
    ; nothing, options::Dict{Symbol, Any} = Dict{Symbol, Any}()) 

    S, ∂S∂C, ϵᵖ, ν, Fᵖ, _, _ = _compute_2nd_PK(mp, C, state, false)

    return S, ∂S∂C, MatHyperElasticPlasticState(ϵᵖ, Fᵖ, ν)
end


function material_response(mp::MatHyperElasticPlastic, C::SymmetricTensor{2,3,T,6}, state::MatHyperElasticPlasticState, Δt=0.0, ::Symbol
    ; nothing, ::Dict{Symbol, Any} = Dict{Symbol, Any}()) 
    
    S, ∂S∂C, ϵᵖ, ν, Fᵖ, g, dgdC = _compute_2nd_PK(mp, C, state, true)

    return S, ∂S∂C, MatHyperElasticPlasticState(ϵᵖ, Fᵖ, ν), MatHyperElasticPlasticExtras(g, dgdC)
end
