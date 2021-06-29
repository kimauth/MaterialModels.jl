

function get_NeoHook_loading()
    _C = range(0.0,  0.005, length=3)
    C = [SymmetricTensor{2,3}((x +1.0, x/10, 0.0, 1.0, 0.0, 1.0)) for x in _C]
    return C
end  

@testset "NeoHook" begin
    m = NeoHook(μ=76.9, λ=115.0)

    loading = get_NeoHook_loading()
    check_jld2(m, loading, "NeoHook")#, debug_print=false, OVERWRITE_JLD2=true)
    check_tangents_AD(m,loading)
end
