function get_StVenant_loading()
    _C = range(0.0,  0.005, length=3)
    C = [SymmetricTensor{2,3}((x + 1.0, x/10, 0.0, 1.0, 0.0, 1.0)) for x in _C]

    return C
end  

@testset "StVenant" begin
    m = StVenant(μ=6.8, λ=62.1)

    loading = get_StVenant_loading()
    check_jld2(m, loading, "StVenant")#, debug_print=false, OVERWRITE_JLD2=true)
end
