const cdf97 = [
 0.091271763114;
−0.057543526229;
−0.591271763114;
 1.11508705;
−0.591271763114;
−0.057543526229;
 0.091271763114
]

"""
    nle_mad(x::AbstractArray{T,4}, f::AbstractVector{T})

Median Absolute Deviation noise-level estimation. f defaults to cdf97 filter.
"""
function nle_mad(x::AbstractArray{T, 4}, f::AbstractVector) where T <: Real
    z = conv(x, reshape(f, :, 1, 1, 1))
    z = conv(z, reshape(f, 1, :, 1, 1))
    # division by 2 is because we're using the 1D wavelet wavelet twice
    return median(abs.(z)) / T(2*0.6745)
end

# this returns the std-dev of circularly symmetric complex normal
function nle_mad(x::AbstractArray{T, 4}, f::AbstractVector) where T <: Complex
    y = cat(reim(x)...; dims=4)
    return nle_mad(y, f) * real(T)(sqrt(2))
end

function ncov_est(x::AbstractArray{T,4}, f::AbstractVector) where T
    # z is considered a vector of noise 
    N1, N2, C, B = size(x)

    xr = reshape(x, N1, N2, 1, C*B)
    zr = conv(xr, reshape(f, :, 1, 1, 1))
    zr = conv(zr, reshape(f, 1, :, 1, 1)) ./ T(2)

    Z = reshape(zr, :, C, B)
    N = size(Z, 1)
    Z = permutedims(Z, (2,1,3)) # C, N, B

    μ = mean(Z, dims=2)
    Zμ = reshape(Z .- μ, C, 1, N*B) 
    Σ = sum(reshape(Zμ ⊠ batched_adjoint(Zμ), C, C, N, B); dims=3)

    # take mean over number of pixels in region
    mask = (sum(abs2, x; dims=3) .> 0)
    N_mask = reshape(sum(mask; dims=(1,2)), 1, 1, B)

    Σ = dropdims(Σ; dims=3) ./ N_mask
    return Σ
end

function sqrt_covmat(Σ::CuArray{T,3}) where T
    # Σ = USUᴴ 
    U, s, _ = svd(Σ)
    sqrt_s = reshape(sqrt.(s), 1, size(s, 1), size(s, 2))
    return U ⊠ batched_adjoint(sqrt_s .* U)
end
function sqrt_covmat(Σ::AbstractArray{T,2}) where T
    # Σ = USUᴴ 
    U, s, _ = svd(Σ)
    sqrt_s = reshape(sqrt.(s), 1, :)
    return U * (sqrt_s .* U)'
end

"""
    whiten(x::AA{T,4}, smaps::AA{T,4}, Σ=ncov_est(x))
`x` is expected to be a (possibly batched) multicoil image domain signal.
`smaps` are the associated sensitivity maps of `x`.
"""
function whiten(x::AbstractArray{T,4}, smaps::AbstractArray{T,4}, Σ::AbstractArray{Ts,3}=ncov_est(x)) where {T,Ts}
    # Σ = USUᴴ 
    U, s, _ = svd(Σ)

    function sqrtΣ(t) 
        t1 = mul_channel(batched_adjoint(U), t)
        t2 = t1 ./ (reshape(sqrt.(s), 1, 1, size(s, 1), size(s, 2)) .+ eps(real(T)))
        t3 = mul_channel(U, t2)
        return t3
    end

    # whiten signal and sensitivity maps
    x_w = sqrtΣ(x) 
    smaps_w = sqrtΣ(smaps) 

    # normalize smaps
    z = sqrt.(sum(abs2, smaps_w; dims=3))
    smaps_w ./= (eps(real(T)) .+ z) 

    # normalize whitened data to have same range as coil-combined input
    β = maximum(abs, sum(conj(smaps) .* x; dims=3); dims=1:2)
    δ = maximum(abs, sum(conj(smaps_w) .* x_w; dims=3); dims=1:2)
    σ = β ./ δ
    x_w .*= σ

    # renormalization factor: return coilcombined whitened image to sensitivity profile of original by multiplying with g
    zinv = (z .> 0) ./ (σ .* z .+ eps(real(T))) 
    σ = (z .> 0) .* σ

    return (data=x_w, smaps=smaps_w, σ=σ, zinv=zinv)
end
whiten(y::AbstractArray{T1,4}, s::AbstractArray{T2,3}, Σ::AbstractArray{T3,2}) where {T1,T2,T3} = whiten(y, s[:,:,:,:], Σ[:,:,:])

function whiten(x::AbstractArray{T,4}, Σ::AbstractArray{Ts,3}=ncov_est(x)) where {T,Ts}
    # Σ = USUᴴ 
    U, s, _ = svd(Σ)
    t1 = mul_channel(batched_adjoint(U), x)
    t2 = t1 ./ (reshape(sqrt.(s), 1, 1, size(s, 1), size(s, 2)) .+ eps(real(T)))
    x_w = mul_channel(U, t2)
    return x_w
end

function whiten(xs::Union{Vector, Tuple}, smaps, Σ::AbstractArray{T, 3}) where T
    # Σ = USUᴴ 
    U, s, _ = svd(Σ)

    function sqrtΣ(t) 
        t1 = mul_channel(batched_adjoint(U), t)
        t2 = t1 ./ (reshape(sqrt.(s), 1, 1, size(s, 1), size(s, 2)) .+ eps(real(T)))
        t3 = mul_channel(U, t2)
        return t3
    end

    # whiten signal and sensitivity maps
    xs_w = [sqrtΣ(x) for x in xs]
    smaps_w = sqrtΣ(smaps) 

    # normalize smaps
    z = sqrt.(sum(abs2, smaps_w; dims=3))
    smaps_w ./= (eps(real(T)) .+ z) 

    # normalize whitened data to have same range as coil-combined input
    S  = Sense(smaps)
    Sw = Sense(smaps_w)

    β = mean(cat([maximum(abs, S'(x); dims=1:2) for x in xs]...; dims=1); dims=1)
    δ = mean(cat([maximum(abs, Sw'(x_w); dims=1:2) for x_w in xs_w]...; dims=1); dims=1)
    σ = β ./ δ
    for x_w in xs_w
        x_w .*= σ
    end

    # renormalization factor: return coilcombined whitened image to sensitivity profile of original by multiplying with g
    zinv = (z .> 0) ./ (σ .* z .+ eps(real(T))) 
    σ = (z .> 0) .* σ

    return (data=xs_w, smaps=smaps_w, σ=σ, zinv=zinv)
end

function whiten(y::AbstractArray{T,5}, s, Σ) where T
    data_vec, smaps, σ, zinv = whiten([selectdim(y, 5, ii) for ii=1:size(y,5)], s, Σ)
    return (data=cat(data_vec...; dims=5), smaps=smaps, σ=σ, zinv=zinv)
end

nle_mad(x::Array{T}) where T   = nle_mad(x, T.(cdf97))
nle_mad(x::CuArray{T}) where T = nle_mad(x, cu(real(T).(cdf97)))
ncov_est(x::Array{T}) where T   = ncov_est(x, T.(cdf97))
ncov_est(x::CuArray{T}) where T = ncov_est(x, cu(real(T).(cdf97)))
