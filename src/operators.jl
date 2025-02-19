module Operators

using Base
using CUDA, cuDNN, NNlib
using LinearAlgebra, FFTW
using MLUtils
using Statistics, Random

export AbstractOperator
export AWGN, AddNoise, Sense, Fourier, Mask
export awgn, acgn, gramian

abstract type AbstractOperator end

struct AddNoise end

struct AdjointOp{S <: AbstractOperator} <: AbstractOperator
    parent::S
end
struct Gramian{S <: AbstractOperator} <: AbstractOperator
    parent::S
end
const AdjointAbsOp = AdjointOp{<:AbstractOperator}
const GramianAbsOp = Gramian{<:AbstractOperator}

Base.adjoint(A::AbstractOperator) = AdjointOp(A)
Base.adjoint(A::AdjointAbsOp) = A.parent
gramian(A::AbstractOperator) = Gramian(A)
# default gramian op
(A::GramianAbsOp)(x) = A.parent'(A.parent(x))

# don't add noise or use rng if your're not supposed to!
(A::AbstractOperator)(x, ::AddNoise) = A(x)
(A::AbstractOperator)(rng::AbstractRNG, x) = A(x)
(A::AbstractOperator)(rng::AbstractRNG, x, ::AddNoise) = A(x)

struct Identity <: AbstractOperator end
(A::Identity)(x) = x
(A::AdjointOp{Identity})(x) = x
(A::Gramian{Identity})(x) = x
tikhonov_solve(::Identity, ρ, x::AbstractArray{T}) where T = x ./ ( one(T) .+ ρ)

struct Null <: AbstractOperator end
(A::Null)(x) = zeros_like(x)
(A::AdjointOp{Null})(x) = zeros_like(x)
(A::Gramian{Null})(x) = zeros_like(x)
function Base.getproperty(A::Null, s::Symbol)
    if s == :noise_level
        return 0f0
    end
    return Base.getfield(A, s)
end

struct ComposedOperator{T<:Tuple} <: AbstractOperator
    ops::T
end
ComposedOperator(ops...) = ComposedOperator(ops)

Base.:(∘)(op1::AbstractOperator, op2::AbstractOperator) = compose(op1, op2)
compose(A::AbstractOperator) = A
compose(::Identity, ::Identity) = Identity()
compose(A::AbstractOperator, ::Identity) = A
compose(::Identity, A::AbstractOperator) = A
compose(A::ComposedOperator, ::Identity) = A
compose(::Identity, A::ComposedOperator) = A
compose(A::AbstractOperator, B::AbstractOperator) = ComposedOperator(A, B)
compose(com::ComposedOperator, A::AbstractOperator) = compose(com.ops..., A)
compose(A::AbstractOperator, com::ComposedOperator) = compose(A, com.ops...)
compose(com1::ComposedOperator, com2::ComposedOperator) = compose(com1.ops..., com2.ops...)
compose(ops...) = compose(compose(ops[1], ops[2]), ops[3:end]...)

function (A::ComposedOperator)(rng::AbstractRNG, x, args...)
    for op in reverse(A.ops)
        x = op(rng, x, args...)
    end
    return x
end
(A::ComposedOperator)(x) = A(Random.GLOBAL_RNG, x)

function (A::AdjointOp{<:ComposedOperator})(rng::AbstractRNG, x, args...)
    for op in A.parent.ops
        x = op'(rng, x, args...)
    end
    return x
end
(A::AdjointOp{<:ComposedOperator})(x) = A(Random.GLOBAL_RNG, x)

mutable struct AWGN{S <: AbstractOperator, T <: Union{<:Real,<:AbstractArray}} <: AbstractOperator
    parent::S
    noise_level::T
end

const AWGNAbsOp = AWGN{<:AbstractOperator}
AWGN(σ) = AWGN(Identity(), σ)
Base.adjoint(A::AWGNAbsOp) = AWGN(A.parent', A.noise_level)
gramian(A::AWGNAbsOp) = AWGN(gramian(A.parent), A.noise_level)
(A::AWGNAbsOp)(x) = A.parent(x)
(A::AWGNAbsOp)(x, ::AddNoise) = awgn(A.parent(x), A.noise_level)
(A::AWGNAbsOp)(rng::AbstractRNG, x, ::AddNoise) = awgn(rng, A.parent(x), A.noise_level)
compose(A::AWGN, B::Identity) = A
compose(A::AWGN, B::AbstractOperator) = AWGN(compose(A.parent, B), A.noise_level)
compose(A::AWGN, B::ComposedOperator) = AWGN(compose(A.parent, B), A.noise_level)
tikhonov_solve(A::AWGNAbsOp, ρ, x) = tikhonov_solve(A.parent, ρ, x)

function Base.getproperty(A::ComposedOperator, s::Symbol)
    if s == :noise_level && any(isa.(A.ops, AWGNAbsOp))
        for op in A.ops
            if hasproperty(op, s)
                return op.noise_level
            end
        end
    end
    if s == :map && any(isa.(A.ops, Sense))
        for op in A.ops
            if hasproperty(op, s)
                return Base.getfield(op, s)
            end
        end
    end
    if (s == :mask || s == :center_mask)  && any(isa.(A.ops, MaskedAbsOp))
        for op in A.ops
            if hasproperty(op, s)
                return Base.getfield(op, s)
            end
        end
    end
    return Base.getfield(A, s)
end

function Base.setproperty!(A::ComposedOperator, s::Symbol, x)
    if s == :noise_level && any(isa.(A.ops, AWGNAbsOp))
        for op in A.ops
            if hasproperty(op, :noise_level)
                return setfield!(op, s, x)
            end
        end
    end
    if s == :map && any(isa.(A.ops, Sense))
        for op in A.ops
            if hasproperty(op, s)
                return setfield!(op, s, x)
            end
        end
    end
    if (s == :mask || s == :center_mask) && any(isa.(A.ops, MaskedAbsOp))
        for op in A.ops
            if hasproperty(op, s)
                return setfield!(op, s, x)
            end
        end
    end
    throw(ArgumentError("Invalid property: $symbol"))
end

mutable struct Masked{S <: AbstractOperator, T1 <: Union{<:Real, <:AbstractArray}, T2 <: Union{Missing, <:Real, <:AbstractArray}} <: AbstractOperator
    parent::S
    mask::T1
    center_mask::T2
end

const MaskedAbsOp = Masked{<:AbstractOperator}
Mask(m, mc) = Masked(Identity(), m, mc)
Mask(m) = Masked(Identity(), m, missing)
Mask(A::AbstractOperator, m, mc) = Masked(A, m, mc)
Mask(A::AbstractOperator, m) = Masked(A, m, missing)
(A::MaskedAbsOp)(x) = A.mask .* A.parent(x)
(A::MaskedAbsOp)(x, ::AddNoise) = A.mask .* A.parent(x, AddNoise())
(A::MaskedAbsOp)(rng::AbstractRNG, x, ::AddNoise) = A.mask .* A.parent(rng, x, AddNoise())
(A::AdjointOp{<:MaskedAbsOp})(x) = A.parent.parent'(A.parent.mask .* x)
(A::Gramian{<:MaskedAbsOp})(x) = A.parent.parent'( A.parent.mask .* A.parent.parent(x))
compose(A::Masked, B::Identity) = A
compose(A::Masked, B::AbstractOperator) = Masked(compose(A.parent, B), A.mask, A.center_mask)
tikhonov_solve(A::Masked{<:AWGNAbsOp}, ρ, x) = tikhonov_solve(Masked(A.parent.parent, A.mask, A.center_mask), ρ, x)

function Base.getproperty(A::Masked{<:AWGNAbsOp}, s::Symbol)
    if s in (:parent, :mask, :center_mask)
        return getfield(A, s)
    end
    if s == :noise_level
        return getfield(A.parent, s)
    end
    return getproperty(A.parent.parent, s)
end

function Base.setproperty!(A::Masked{<:AWGNAbsOp}, s::Symbol, x)
    if s in (:parent, :mask, :center_mask)
        return setfield!(A, s, x)
    end
    if s == :noise_level 
        return setfield!(A.parent, s, x)
    end
    return setproperty!(A.parent.parent, s, x)
end

struct PFourier{N,F,B} <: AbstractOperator
    fwd::F
    bwd::B
end
function PFourier(x::AbstractArray{T,N}) where {T <: Complex, N}
    fwd = plan_fft(x, 1:N-2)
    y = fwd*x
    bwd = plan_ifft(y, 1:N-2)
    PFourier{N-2, typeof(fwd), typeof(bwd)}(fwd, bwd)
end
function PFourier(x::AbstractArray{T,N}) where {T <: Real, N}
    fwd = plan_rfft(x, 1:N-2)
    y = fwd*x
    bwd = plan_irfft(y, size(y,1), 1:N-2)
    PFourier{N-2, typeof(fwd), typeof(bwd)}(fwd, bwd)
end
(A::PFourier{N,F,B})(x) where {N,F,B} = ifftshift(A.fwd * fftshift(x, 1:N), 1:N)
(A::AdjointOp{PFourier{N,F,B}})(x) where {N,F,B} = ifftshift(A.parent.bwd * fftshift(x, 1:N), 1:N)
(A::Gramian{PFourier})(x) = x

struct Fourier{N} <: AbstractOperator end
(A::Fourier{N})(x) where N = ifftshift(fft(fftshift(x, 1:N), 1:N), 1:N)
(A::AdjointOp{Fourier{N}})(x) where N = ifftshift(ifft(fftshift(x, 1:N), 1:N), 1:N)
(A::Gramian{Fourier})(x) = x
tikhonov_solve(::Fourier, ρ, x) = tikhonov_solve(Identity(), ρ, x)

function tikhonov_solve(A::Masked{Fourier}, ρ, x::AbstractArray{T,N}) where {T,N}
    F = Fourier(N-2)
    Fx = F(x)
    return F'(@. Fx / (abs2(A.mask) + ρ))
end

mutable struct Sense{T<:AbstractArray} <: AbstractOperator 
    map::T
end
(S::Sense)(x::AbstractArray) = S.map .* x
(S::AdjointOp{<:Sense})(x::AbstractArray) = sum(conj(S.parent.map) .* x; dims=3)
(S::Gramian{<:Sense})(x::AbstractArray) = sum(abs2, S.parent.map; dims=3) .* x

struct Resample <: AbstractOperator 
    scale::Int
end
Downsample(factor) = Resample(1/factor)
Upsample(factor) = Resample(factor)

upsample(x::AbstractArray{T, 4}, s) where T = NNlib.upsample_bilinear(x, s)
function upsample(x::AbstractArray{<:Complex, 4}, s)
    xr, xi = reim(x)
    yr = NNlib.upsample_bilinear(real(xr), s) |> real
    yi = NNlib.upsample_bilinear(real(xi), s) |> real
    return yr + 1im*yi
end

(H::Resample)(x::AbstractArray{T,4}) where T = upsample(x, (2, 2))
(H::AdjointOp{<:Resample})(x::AbstractArray{T,4}) where T = upsample(x, (0.5, 0.5))

function galerkin(A::AbstractOperator) 
    B = Resample(2)
    return A ∘ B
end
galerkin(A::Identity) = A 
galerkin(A::AWGN) = A 

#==========================================
     Additive White Gaussian Noise
==========================================#
awgn(rng::AbstractRNG, x, σ) =  x + σ .* randn!(rng, similar(x))
awgn(x, σ) = x + σ .* randn!(similar(x))

#==========================================
Additive Correlated(/Colored) Gaussian Noise
==========================================#
function acgn(rng::AbstractRNG, x::CuArray{T1,N}, L::AbstractArray{T2,3}) where {T1, T2, N}
    xr = reshape(x, :, size(x)[N-1:N]...)
    xr = permutedims(xr, (2,1,3))
    yr = xr + (L ⊠ randn!(rng, similar(xr)))
    yr = permutedims(yr, (2,1,3))
    return reshape(yr, size(x))
end
acgn(x, L) = acgn(Random.GLOBAL_RNG, x, L)

function rand_sqrtcov(rng::AbstractRNG, T::Type, channels::Int, batch::Int, σ, b)
    mask = Diagonal(CUDA.ones(channels)) .> 0
    A = (1f0 .- mask) .* b .* (2f0 .* rand!(rng, CUDA.ones(T, channels, channels, batch)) .- 1f0) ./ (channels-1) .+  mask .* σ  
    return A
end
rand_sqrtcov(chan::Int, batch::Int, σ, b) = rand_sqrtcov(Float32, chan, batch, σ, b)

function rand_sqrtcov(rng::AbstractRNG, x::AbstractArray{T}, σdiag, σjitter, σcorr) where T
    σ = σdiag .+ σjitter .* randn!(rng, similar(σdiag))
    rand_sqrtcov(rng, T, size(x, 3), size(x, 4), σ, σcorr)
end
rand_sqrtcov(x::AbstractArray, args...) = rand_sqrtcov(Random.GLOBAL_RNG, x, args...) 

function rand_range(rng::AbstractRNG, batch::AbstractArray{T, N}, (a, b)::Union{Vector, Tuple}) where {T, N}
    return a .+ (b-a).*rand!(rng, similar(batch, real(T), (ones(Int, N-1)..., size(batch, N))))
end
rand_range(batch, t) = rand_range(Random.GLOBAL_RNG, batch, t)

rand_range(rng::AbstractRNG, batch, σ::Real) = rand_range(rng, batch, (σ, σ))

end;
