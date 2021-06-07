# General residual function 
function residual(X::ChabocheResidual{NKin_R}, material::Chaboche, old::ChabocheState, ϵ) where{NKin_R}
    Δλ = X.λ - old.λ
    
    σ_vm = vonMisesDev(X.σ_red_dev)
    ν = X.σ_red_dev * ((3/2)*σ_vm)

    Φ = yieldCriterion(material, X.σ_red_dev, X.λ)

    σ_dev = calc_sigma_dev(material.elastic, old, ϵ, ν, Δλ)
    
    if NKin_R > 0
        β_hat0 = σ_dev - X.σ_red_dev - sum(X.β1)
        β_hat1 = X.β1
        β1 = ntuple(i->old.β[i+1] + Δλ * KinematicEvolution(material.kinematic[i+1], ν, β_hat1[i]), Val{NKin_R}())
    else
        β_hat0 = σ_dev - X.σ_red_dev
    end

    β0 = old.β[1] + Δλ * KinematicEvolution(material.kinematic[1], ν, β_hat0)

    if NKin_R > 0
        σ_red_dev = σ_dev - β0 - sum(β1)
        R = ChabocheResidual(Φ, X.σ_red_dev-σ_red_dev, ntuple(i->β1[i]-β_hat1[i], Val{NKin_R}()))
    else
        σ_red_dev = σ_dev - β0
        R = ChabocheResidual(Φ, X.σ_red_dev-σ_red_dev)
    end
    return R
end

# Specialized for only one backstress (NKin_R=0)
function residual(X::ChabocheResidual{0}, material::Chaboche, old::ChabocheState, ϵ)
    
    σ_vm = vonMisesDev(X.σ_red_dev)
    ν = X.σ_red_dev * ((3/2)*σ_vm)
    Φ = yieldCriterion(material, σ_vm, X.λ)

    σ_dev = calc_sigma_dev(material.elastic, old, ϵ, ν, X.λ - old.λ)
    β_hat0 = σ_dev - X.σ_red_dev
    
    β0 = old.β[1] + (X.λ - old.λ) * KinematicEvolution(material.kinematic[1], ν, β_hat0)
    
    σ_red_dev = σ_dev - β0
    
    return ChabocheResidual(Φ, X.σ_red_dev-σ_red_dev)
end