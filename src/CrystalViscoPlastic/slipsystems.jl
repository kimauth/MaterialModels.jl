abstract type AtomicStructure end

struct BCC <: AtomicStructure end
struct FCC <: AtomicStructure end

# TODO: introduce type Slipsystem instead of using NTuple

#slip planes and slip directions for BCC
function _get_slipsystems(::FCC)
    slip_planes = [ [ 1., 1., 1.],
                    [ 1., 1.,-1.],
                    [ 1.,-1.,-1.],
                    [ 1.,-1., 1.]]
    slip_directions = [ ([ 1., 0.,-1.], [ 0., 1.,-1.], [ 1.,-1., 0.]);
                        ([ 0., 1., 1.], [ 1., 0., 1.], [ 1.,-1., 0.]);
                        ([ 1., 0., 1.], [ 0., 1.,-1.], [ 1., 1., 0.]);
                        ([ 0., 1., 1.], [ 1., 0.,-1.], [ 1., 1., 0.]) ]
    return slip_planes, slip_directions
end

function _get_slipsystems(::BCC)
    slip_planes = [ [ 1., 1., 0.],
                    [ 1.,-1., 0.],
                    [ 1., 0., 1.],
                    [-1., 0.,-1.],
                    [ 0., 1., 1.],
                    [ 0., 1.,-1.]]
    slip_directions = [ ([ 1.,-1., 1.], [ 1.,-1.,-1.]);
                        ([ 1., 1., 1.], [ 1., 1.,-1.]);
                        ([-1., 1., 1.], [-1.,-1., 1.]);
                        ([ 1., 1., 1.], [ 1.,-1., 1.]);
                        ([ 1., 1.,-1.], [-1., 1.,-1.]);
                        ([-1., 1., 1.], [ 1., 1., 1.]) ]
    return slip_planes, slip_directions
end

function slipsystems(as::AtomicStructure, R::Rotations.Rotation)
    # generate rotated slip systems
    slip_systems = NTuple{2, Tensor{1,3}}[]
    slip_planes, slip_directions = _get_slipsystems(as)
    for (i, sp) in enumerate(slip_planes)
        for sd in slip_directions[i]
            push!(slip_systems, (Tensor{1,3}(R*sp), Tensor{1,3}(R*sd)))
        end
    end

    normalized_slip_systems = Vector{NTuple{2, Tensor{1,3,Float64,3}}}(undef, length(slip_systems))
    for i=1:length(slip_systems)
        nss = (slip_systems[i][1] / norm(slip_systems[i][1]), slip_systems[i][2] / norm(slip_systems[i][2]))
        normalized_slip_systems[i] = nss
    end
    return normalized_slip_systems
end
