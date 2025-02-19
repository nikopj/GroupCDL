#==============================================================================
===============================================================================
===============================================================================
                       NATURAL IMAGE SAMPLES
===============================================================================
===============================================================================
==============================================================================#

abstract type NoisyNaturalImage <: NoisyImage end
noise_level(s::NoisyNaturalImage) = s.noise_level

#==============================================================================
                           GT NATURAL IMAGE
==============================================================================#
struct GTNaturalImage <: Image
    image::AbstractArray
    path
end
image(s::GTNaturalImage) = s.image

function Adapt.adapt_structure(to, sample::GTNaturalImage)
    image = Adapt.adapt_structure(to, sample.image)
    GTNaturalImage(image, sample.path)
end
function CUDA.unsafe_free!(s::GTNaturalImage)
    CUDA.unsafe_free!(s.image)
end
function MLUtils.batch(xs::Vector{<:GTNaturalImage})
    GTNaturalImage(
        MLUtils.batch([x.image for x in xs]), 
        MLUtils.batch([x.path for x in xs]),
    )
end

const NaturalImage = Union{GTNaturalImage, NoisyNaturalImage}

maxval(s::NaturalImage) = 1f0
name(s::NaturalImage) = (s.path isa Vector) ? first(s.path) : s.path 

#==============================================================================
                     PAIRED NATURAL (NOISY) IMAGE
==============================================================================#
struct PairedNaturalImage <: NoisyNaturalImage
    image::AbstractArray
    noisy::AbstractArray
    noise_level::AbstractArray
    path
end
input(s::PairedNaturalImage) = s.noisy
target(s::PairedNaturalImage) = s.image
reference(s::PairedNaturalImage) = target(s)

function Adapt.adapt_structure(to, sample::PairedNaturalImage)
    image = Adapt.adapt_structure(to, sample.image)
    noisy = Adapt.adapt_structure(to, sample.noisy)
    σ     = Adapt.adapt_structure(to, sample.noise_level)
    PairedNaturalImage(image, noisy, σ, sample.path)
end
function CUDA.unsafe_free!(s::PairedNaturalImage)
    CUDA.unsafe_free!(s.image)
    CUDA.unsafe_free!(s.noisy)
    CUDA.unsafe_free!(s.noise_level)
end
function MLUtils.batch(xs::Vector{<:PairedNaturalImage})
    PairedNaturalImage(
        MLUtils.batch([x.image for x in xs]), 
        MLUtils.batch([x.noisy for x in xs]), 
        MLUtils.batch([x.noise_level for x in xs]), 
        MLUtils.batch([x.path for x in xs]),
    )
end

#==============================================================================
                     TRIPLE NATURAL (NOISY) IMAGE
==============================================================================#
struct TripleNaturalImage <: NoisyNaturalImage
    image::AbstractArray
    noisy_a::AbstractArray
    noisy_b::AbstractArray
    noise_level::AbstractArray
    path
end
input(s::TripleNaturalImage) = s.noisy_a
target(s::TripleNaturalImage) = s.noisy_b
reference(s::TripleNaturalImage) = s.image

function Adapt.adapt_structure(to, sample::TripleNaturalImage)
    image = Adapt.adapt_structure(to, sample.image)
    noisy_a = Adapt.adapt_structure(to, sample.noisy_a)
    noisy_b = Adapt.adapt_structure(to, sample.noisy_b)
    σ     = Adapt.adapt_structure(to, sample.noise_level)
    TripleNaturalImage(image, noisy_a, noisy_b, σ, sample.path)
end
function CUDA.unsafe_free!(s::TripleNaturalImage)
    CUDA.unsafe_free!(s.image)
    CUDA.unsafe_free!(s.noisy_a)
    CUDA.unsafe_free!(s.noisy_b)
    CUDA.unsafe_free!(s.noise_level)
end
function MLUtils.batch(xs::Vector{<:TripleNaturalImage})
    TripleNaturalImage(
        MLUtils.batch([x.image for x in xs]), 
        MLUtils.batch([x.noisy_a for x in xs]), 
        MLUtils.batch([x.noisy_b for x in xs]), 
        MLUtils.batch([x.noise_level for x in xs]), 
        MLUtils.batch([x.path for x in xs]),
    )
end

