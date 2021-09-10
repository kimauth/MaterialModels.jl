
#utility function for checking checksums of materials...
function check_jld2(material::MaterialModels.AbstractMaterial, loading::Vector, filename::String; 
                        OVERWRITE_JLD2::Bool = false, debug_print::Bool = false)

    state = initial_material_state(material)
    stresses = []; tangents = [];
    for (i,load) in enumerate(loading)
        stress, tangent, state = material_response(material, load, state)

        #For user to check that the material acutally enters plastic state
        # and is not only testing the elastic part...
        if debug_print 
            println("Iteration: $i")
            for symbol in fieldnames(typeof(state))
                println("$symbol: $(getproperty(state, symbol))")
            end
        end
        push!(stresses, stress)
        push!(tangents, tangent)
    end

    jld2_file = joinpath(@__DIR__, "jld2_files", string(filename, ".jld2"))
    if OVERWRITE_JLD2
        jldsave(jld2_file; stresses_jld2=stresses, tangents_jld2=tangents)
    else
        stresses_jld2, tangents_jld2 = load(jld2_file, "stresses_jld2", "tangents_jld2")
        @test stresses ≈ stresses_jld2
        @test tangents ≈ tangents_jld2
    end

end

function check_tangents_AD(material::MaterialModels.AbstractMaterial,loading::Vector)
    state = initial_material_state(material)
    for load in loading
        stress, tangent, state = material_response(material, load, state)
        ∂²Ψ∂C², ∂Ψ∂C, _ =  hessian(C -> MaterialModels.ψ(material, C), load, :all)
        stress_AD = 2*∂Ψ∂C
        tangent_AD = 4*∂²Ψ∂C²
        @test stress ≈ stress_AD
        @test tangent ≈ tangent_AD
    end
end
