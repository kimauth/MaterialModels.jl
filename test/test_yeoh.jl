function get_Yeoh_loading()
    _C = range(0.0,  0.005, length=3)
    C = [SymmetricTensor{2,3}((x + 1.0, x/10, 0.0, 1.0, 0.0, 1.0)) for x in _C]

    return C
end  

@testset "Yeoh" begin
    m = Yeoh(μ = 6.8, λ=62.1, c₂ = -6.8/10, c₃ = 6.8/30)

    loading = get_Yeoh_loading()
    check_jld2(m, loading, "Yeoh")#, debug_print=false, OVERWRITE_JLD2=true)
    check_tangents_AD(m,loading)
end
