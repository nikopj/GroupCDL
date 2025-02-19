using NNlib
using ..GroupCDL: whiten, mul_channel
using ..Operators

#==============================================================================
                           GT MODL BRAIN IMAGE
==============================================================================#
abstract type AbstractMoDLImage <: Image end
abstract type NoisyMoDLImage{cc} <: NoisyImage end # cc == coilcombined noisy data

Samples.maxval(s::AbstractMoDLImage) = 1f0
Samples.name(s::AbstractMoDLImage) = "slice_$((s.slicenum isa Vector) ? first(s.slicenum) : s.slicenum)"

is_coilcombined(s::NoisyMoDLImage{cc}) where cc = cc
is_multicoil(s::NoisyMoDLImage{cc}) where cc = !cc
Samples.noise_level(s::NoisyMoDLImage) = s.L

struct GTMoDLImage <: AbstractMoDLImage 
    image::AbstractArray # coil-combined 
    smaps::AbstractArray
    slicenum
end
image(s::GTMoDLImage) = s.image

const MoDLImage = Union{GTMoDLImage, NoisyMoDLImage}
Operators.Sense(s::MoDLImage) = Sense(s.smaps)

function Adapt.adapt_structure(to, sample::GTMoDLImage)
    image = Adapt.adapt_structure(to, sample.image)
    smaps = Adapt.adapt_structure(to, sample.smaps)
    GTMoDLImage(image, smaps, sample.slicenum)
end
function CUDA.unsafe_free!(s::GTMoDLImage)
    CUDA.unsafe_free!(s.image)
    CUDA.unsafe_free!(s.smaps)
end
function MLUtils.batch(xs::Vector{<:GTMoDLImage})
    GTMoDLImage(
        MLUtils.batch([x.image for x in xs]), 
        MLUtils.batch([x.smaps for x in xs]), 
        MLUtils.batch([x.slicenum for x in xs]),
    )
end

#==============================================================================
                        PAIRED NOISY MODL BRAIN IMAGE
==============================================================================#

struct PairedMoDLImage{cc} <: NoisyMoDLImage{cc}
    image::AbstractArray # coil-combined
    noisy::AbstractArray # multicoil or coilcombined
    smaps::AbstractArray
    L::AbstractArray     # sqrt-covmat or noise-level map
    slicenum
    function PairedMoDLImage(image, noisy, smaps, L, sn)
        cc = size(noisy, 3) == 1
        new{cc}(image, noisy, smaps, L, sn)
    end
end
input(s::PairedMoDLImage) = s.noisy
target(s::PairedMoDLImage) = s.image
reference(s::PairedMoDLImage) = target(s)

function Adapt.adapt_structure(to, sample::PairedMoDLImage)
    image = Adapt.adapt_structure(to, sample.image)
    noisy = Adapt.adapt_structure(to, sample.noisy)
    smaps = Adapt.adapt_structure(to, sample.smaps)
    L     = Adapt.adapt_structure(to, sample.L)
    PairedMoDLImage(image, noisy, smaps, L, sample.slicenum)
end
function CUDA.unsafe_free!(s::PairedMoDLImage)
    CUDA.unsafe_free!(s.image)
    CUDA.unsafe_free!(s.noisy)
    CUDA.unsafe_free!(s.smaps)
    CUDA.unsafe_free!(s.L)
end
function MLUtils.batch(xs::Vector{<:PairedMoDLImage})
    PairedMoDLImage(
        MLUtils.batch([x.image for x in xs]), 
        MLUtils.batch([x.noisy for x in xs]), 
        MLUtils.batch([x.smaps for x in xs]), 
        MLUtils.batch([x.L for x in xs]), 
        MLUtils.batch([x.slicenum for x in xs]),
    )
end

#==============================================================================
                        TRIPLE NOISY MODL BRAIN IMAGE
==============================================================================#

struct TripleMoDLImage{cc} <: NoisyMoDLImage{cc}
    image::AbstractArray   # coil-combined
    noisy_a::AbstractArray # multicoil or coil-combined
    noisy_b::AbstractArray # multicoil or coil-combined
    smaps::AbstractArray
    L::AbstractArray       # sqrt-covmat or noise-level map
    slicenum
    function TripleMoDLImage(image, noisy_a, noisy_b, smaps, L, sn)
        cc = size(noisy_a, 3) == 1
        new{cc}(image, noisy_a, noisy_b, smaps, L, sn)
    end
end
input(s::TripleMoDLImage) = s.noisy_a
target(s::TripleMoDLImage) = s.noisy_b
reference(s::TripleMoDLImage) = s.image

function Adapt.adapt_structure(to, sample::TripleMoDLImage)
    image = Adapt.adapt_structure(to, sample.image)
    noisy_a = Adapt.adapt_structure(to, sample.noisy_a)
    noisy_b = Adapt.adapt_structure(to, sample.noisy_b)
    smaps   = Adapt.adapt_structure(to, sample.smaps)
    L     = Adapt.adapt_structure(to, sample.L)
    TripleMoDLImage(image, noisy_a, noisy_b, smaps, L, sample.slicenum)
end
function CUDA.unsafe_free!(s::TripleMoDLImage)
    CUDA.unsafe_free!(s.image)
    CUDA.unsafe_free!(s.noisy_a)
    CUDA.unsafe_free!(s.noisy_b)
    CUDA.unsafe_free!(s.smaps)
    CUDA.unsafe_free!(s.L)
end
function MLUtils.batch(xs::Vector{<:TripleMoDLImage})
    TripleMoDLImage(
        MLUtils.batch([x.image for x in xs]), 
        MLUtils.batch([x.noisy_a for x in xs]), 
        MLUtils.batch([x.noisy_b for x in xs]), 
        MLUtils.batch([x.smaps for x in xs]), 
        MLUtils.batch([x.L for x in xs]), 
        MLUtils.batch([x.slicenum for x in xs]),
    )
end

#==============================================================================
                       COIL-COMBINE + WHITEN-CC
==============================================================================#
function whitencc(sample::PairedMoDLImage{false}; precomp=false)
    Σ = sample.L ⊠ batched_adjoint(sample.L)
    nw, smaps, σ, zcomp = whiten(sample.noisy, sample.smaps, Σ)
    R = Sense(smaps)
    if precomp
        nw .*= zcomp
        σ  .*= zcomp
        zcomp = 1f0
    end
    PairedMoDLImage(sample.image, R'(nw), smaps, σ, sample.slicenum), zcomp
end

function whitencc(sample::TripleMoDLImage{false}; precomp=false)
    Σ = sample.L ⊠ batched_adjoint(sample.L)
    data, smaps, σ, zcomp = whiten((sample.noisy_a, sample.noisy_b), sample.smaps, Σ)
    R = Sense(smaps)
    nw_a, nw_b = data
    if precomp
        nw_a .*= zcomp
        nw_b .*= zcomp
        σ    .*= zcomp
        zcomp = 1f0
    end
    TripleMoDLImage(sample.image, R'(nw_a), R'(nw_b), smaps, σ, sample.slicenum), zcomp
end

function coilcombine(sample::PairedMoDLImage{false})
    σ = sqrt.(max.(0f0, real(sum(abs2, mul_channel(sample.L, sample.smaps); dims=3))))
    S = Sense(sample.smaps)
    PairedMoDLImage(sample.image, S'(sample.noisy), sample.smaps, σ, sample.slicenum)
end

function coilcombine(sample::TripleMoDLImage{false})
    σ = sqrt.(max.(0f0, real(sum(abs2, mul_channel(sample.L, sample.smaps); dims=3))))
    S = Sense(sample.smaps)
    TripleMoDLImage(sample.image, S'(sample.noisy_a), S'(sample.noisy_b), sample.smaps, σ, sample.slicenum)
end

