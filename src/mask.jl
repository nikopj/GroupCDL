"""
    centercrop(x, y)

Center crop x and y to the minimum either size in the first two dimensions.
"""
function centercrop(x::AbstractArray, y::AbstractArray)
    N1 = min(size(x, 1), size(y, 1))
    N2 = min(size(x, 2), size(y, 2))

    p1 = size(x, 1) - N1
    p2 = size(x, 2) - N2
    p1l, p1r = cld(p1, 2), fld(p1, 2)
    p2l, p2r = cld(p2, 2), fld(p2, 2)
    xc = selectdim(x,  1, 1 + p1l : size(x,1) - p1r)
    xc = selectdim(xc, 2, 1 + p2l : size(x,2) - p2r)

    p1 = size(y, 1) - N1
    p2 = size(y, 2) - N2
    p1l, p1r = cld(p1, 2), fld(p1, 2)
    p2l, p2r = cld(p2, 2), fld(p2, 2)
    yc = selectdim(y,  1, 1 + p1l : size(y,1) - p1r)
    yc = selectdim(yc, 2, 1 + p2l : size(y,2) - p2r)

    return xc, yc
end

function centercrop(x::AbstractArray, N1::Int, N2::Int)
    p1 = size(x, 1) - N1
    p2 = size(x, 2) - N2
    p1l, p1r = cld(p1, 2), fld(p1, 2)
    p2l, p2r = cld(p2, 2), fld(p2, 2)
    xc = selectdim(x,  1, 1 + p1l : size(x,1) - p1r)
    xc = selectdim(xc, 2, 1 + p2l : size(x,2) - p2r)
    return xc
end
centercrop(x, N::Int) = centercrop(x, N, N)

function generate_center_mask(N::Integer, center_frac)
    center_mask = zeros(Bool, N)
    Nc  = round(Int, N*center_frac)
    if Nc % 2 == 0
        Nc = Nc - 1
    end
    pad = (N - Nc + 1) ÷ 2
    center_mask[pad+1:pad + Nc] .= true
    return center_mask
end

function generate_random_mask(N::Integer, B::Integer=1, accel::Integer=2, center_frac=0; readout_dir=:horizontal, rng=Random.GLOBAL_RNG, return_separate_mask=false)
    accel_mask = rand(rng, N, B) .< (1/accel - center_frac)
    center_mask = generate_center_mask(N, center_frac)
    m = accel_mask .|| center_mask

    (m, am, cm) = if readout_dir in (:horizontal, "horizontal", "h", :h)
        (reshape(m,           N, 1, 1, B), 
         reshape(accel_mask,  N, 1, 1, B),
         reshape(center_mask, N, 1, 1, 1))
    elseif readout_dir in (:vertical, "vertical", "v", :v)
        (reshape(m,           1, N, 1, B), 
         reshape(accel_mask,  1, N, 1, B),
         reshape(center_mask, 1, N, 1, 1))
    else 
        throw(ErrorException("readout_dir $readout_dir not implemented."))
    end
    if return_separate_mask
        return am, cm
    end
    return m
end

function generate_uniform_mask(N::Integer, B::Integer=1; accel::Integer=2, center_frac=0, offset=missing, readout_dir=:horizontal, return_separate_mask=false, adjust_accel=true, rng=Random.GLOBAL_RNG)
    Nc = round(Int, N*center_frac)
    adj_accel = adjust_accel ? round(Int, (N - Nc) / (N/accel - Nc)) : accel
    accel_mask = zeros(Bool, N, B)
    offset = ismissing(offset) ? rand(rng, 1:adj_accel, B) : repeat([offset], B)
    for b=1:B
        accel_mask[offset[b] : adj_accel : end, b] .= true
    end
    center_mask = generate_center_mask(N, center_frac)
    m = accel_mask .|| center_mask

    (m, am, cm) = if readout_dir in (:horizontal, "horizontal", "h", :h)
        (reshape(m,           N, 1, 1, B), 
         reshape(accel_mask,  N, 1, 1, B),
         reshape(center_mask, N, 1, 1, 1))
    elseif readout_dir in (:vertical, "vertical", "v", :v)
        (reshape(m,           1, N, 1, B), 
         reshape(accel_mask,  1, N, 1, B),
         reshape(center_mask, 1, N, 1, 1))
    else 
        throw(ErrorException("readout_dir $readout_dir not implemented."))
    end
    if return_separate_mask
        return am, cm
    end
    return m
end

function generate_mask(N, B; type=:random, accel=2, center_frac=0, readout_dir=:horizontal, return_separate_mask=false, rng=Random.GLOBAL_RNG, offset=missing)
    if type in (:random, "random")
        return generate_random_mask(N, B, accel, center_frac; readout_dir=readout_dir, rng=rng, return_separate_mask=return_separate_mask)
    elseif type in (:uniform, "uniform")
        return generate_uniform_mask(N, B; accel=accel, center_frac=center_frac, readout_dir=readout_dir, rng=rng, return_separate_mask=return_separate_mask, offset=offset)
    elseif type in (:identity, "identity")
        return [1f0;;;;]
    end
    throw(ArgumentError("expected type to be :uniform or :random, but got $type."))
end

