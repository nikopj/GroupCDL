abstract type AbstractPreprocess end

struct IdentityPreprocess <: AbstractPreprocess end
struct ImagePreprocess <: AbstractPreprocess 
    stride::Int
    resize_noise::Bool
end
struct KSpacePreprocess <: AbstractPreprocess 
    stride::Int
    resize_noise::Bool
end

function calcpad(N::Int, s::Int)
    p = s*cld(N, s) - N
    return cld(p, 2), fld(p, 2)
end
function calcpad(M::Int, N::Int, s::Int)
    return Tuple([calcpad(M, s)..., calcpad(N, s)...])
end
CRC.@non_differentiable calcpad(::Any...)

function unpad(x::AbstractArray, pad::NTuple{4, Int}) 
    return x[begin+pad[1]:end-pad[2], begin+pad[3]:end-pad[4], :, :]
end

function preprocess(P::ImagePreprocess, y, H::AbstractOperator) 
    ỹ = H'(y)
    pad = calcpad(size(y,1), size(y,2), P.stride)
    yp = pad_reflect(ỹ, pad, dims=(1,2))
    μ = mean(yp, dims=(1,2,3))
    return yp .- μ, (μ, pad)
end

function preprocess(P::ImagePreprocess, y, H::AWGN) 
    ỹ = H'(y)
    pad = calcpad(size(y,1), size(y,2), P.stride)
    yp = pad_reflect(ỹ, pad, dims=(1,2))
    # Zygote.@ignore if P.resize_noise
    if P.resize_noise
        H.noise_level = upsample_bilinear(H.noise_level; size=(cld(size(yp, 1), P.stride), cld(size(yp, 2), P.stride)))
    end
    μ = mean(yp, dims=(1,2,3))
    return yp .- μ, (μ, pad)
end

function NNlib.upsample_nearest(x::AbstractArray{T,N}, scales::NTuple{S, <:Real}) where {T,N,S}
    scales_up, scales_down = rationalize.(scales) |> r->(numerator.(r), denominator.(r))
    up_x = upsample_nearest(x, scales_up)
    down_indices = ntuple(N) do d
        d > S ? Colon() : (1 : scales_down[d] : size(up_x,d))
    end
    return up_x[down_indices...]
end

function preprocess(P::KSpacePreprocess, y, H::AbstractOperator) 
    x = H'(y)
    pad = calcpad(size(x,1), size(x,2), P.stride)
    xp = pad_reflect(x, pad, dims=(1,2))
    Zygote.@ignore begin
        if size(H.mask, 2) == 1
            H.mask = upsample_nearest(H.mask, (size(xp,1)/size(x,1), 1.0))
        end
        if size(H.mask, 1) == 1
            H.mask = upsample_nearest(H.mask, (1.0, size(xp,2)/size(x,2)))
        end
    end
    Zygote.@ignore H.map = pad_reflect(H.map, pad, dims=(1,2))
    Zygote.@ignore if P.resize_noise
        σ = H.noise_level
        σ = upsample_bilinear(σ; size=(cld(size(xp, 1), P.stride), cld(size(xp, 2), P.stride)))
        H.noise_level = σ
    end
    μ = mean(xp, dims=(1,2))
    return xp .- μ, (μ, pad)
end

preprocess(::IdentityPreprocess, x, ::AbstractOperator) = x, (0,0)
postprocess(::IdentityPreprocess, x, ::Tuple) = x

function postprocess(::Union{ImagePreprocess, KSpacePreprocess}, x, (μ, pad)::Tuple) 
    return unpad(x .+ μ, pad)
end
