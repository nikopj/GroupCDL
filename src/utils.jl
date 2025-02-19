totuple(a::Tuple) = a
totuple(a::Vector) = (a[1], a[2])
totuple(a::Number) = (a, a)

function mul_channel(A::AbstractArray{Ta,3}, b::AbstractArray{Tb,N}) where {Ta, Tb, N}
    @assert size(A,2) == size(b, 3)

    B = reshape(b, :, size(b)[N-1:N]...) 
    B = permutedims(B, (2,1,3))

    C = A ⊠ B

    c = permutedims(C, (2,1,3))
    return reshape(c, size(b)[1:N-2]..., size(A,1), :)
end
mul_channel(A::AbstractArray{T,2}, b) where T = mul_channel(A[:,:,:], b)

function center_crop(x, N1, N2)
    sx, sy = size(x)[1:2] .÷ 2

    p1, p2 = fld(N1, 2), cld(N1, 2)
    y = selectdim(x, 1, sx-p1+1:sx+p2)

    p1, p2 = fld(N2, 2), cld(N2, 2)
    y = selectdim(y, 2, sy-p1+1:sy+p2)

    return y
end
center_crop(x, N) = center_crop(x, N, N)

function calc_maxpad(M, N)
    m = 2max(M, N)

    p1 = m - M
    p1l, p1r = fld(p1, 2), cld(p1, 2)

    p2 = m - N
    p2l, p2r = fld(p2, 2), cld(p2, 2)

    return (p1l, p1r, p2l, p2r)
end
CRC.@non_differentiable calc_maxpad(::Any...)

function pad2square(x::AbstractArray)
    pad = calc_maxpad(size(x,1), size(x,2))
    return pad_zeros(x, pad; dims=(1,2)), pad
end

function rss(x)
    sqrt.(sum(abs2, x; dims=3))
end

