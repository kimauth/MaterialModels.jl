
#utility function for checking checksums of materials...
function check_checksum(material::MaterialModels.AbstractMaterial, loading::Vector, filename::String; 
                        OVERWRITE_CHECKSUMS::Bool = false, debug_print::Bool = false)

    #  
    checksums_file = joinpath(dirname(@__FILE__), "checksums", string(filename,".sha1"))

    if OVERWRITE_CHECKSUMS
        csio = open(checksums_file, "w")
    else
        csio = open(checksums_file, "r")
    end

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

    checkhash1 = string(hash(stresses))
    checkhash2 = string(hash(tangents))
    if OVERWRITE_CHECKSUMS
        write(csio, checkhash1, "\n")
        write(csio, checkhash2, "\n")
        close(csio)
    else
        @test chomp(readline(csio)) == checkhash1
        @test chomp(readline(csio)) == checkhash2
    end

end
