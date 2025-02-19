#==============================================================================
===============================================================================
===============================================================================
                       MRI KSPACE SAMPLES
===============================================================================
===============================================================================
==============================================================================#

abstract type KSpace <: AbstractSample end
abstract type ObservedKSpace <: KSpace end

kspace(s::KSpace) = s.kspace
input(s::KSpace) = kspace(s)
target(::KSpace) = missing
reference(::KSpace) = missing
maxval(s::KSpace) = maximum(rss(kspace(s)))
name(s::KSpace) = s.name
noise_level(::KSpace) = missing
obs_operator(::KSpace) = missing

const AA = AbstractArray
const MissingAA = Union{Missing, AbstractArray}

obs_operator(s::ObservedKSpace) = (Mask(s.mask, s.center_mask) ∘ AWGN(noise_level(s)) ∘ Fourier{2}() ∘ Sense(s.sense_maps))
noise_level(s::ObservedKSpace) = sqrt.(sum(abs2, mul_channel(s.L, s.sense_maps); dims=3))

struct MaskedKSpace{Tk, Tm, Tc, Ts, Tσ} <: ObservedKSpace 
    kspace::Tk
    mask::Tm
    center_mask::Tc
    sense_maps::Ts
    L::Tσ          # sqrt-covmat or noise-level map
    name
end
input(s::MaskedKSpace) = s.kspace
noise_level(s::MaskedKSpace{Tk, Tm, Tc, Ts, <:AbstractArray{T,4}}) where {Tk, Tm, Tc, Ts, T} = s.L

function Adapt.adapt_structure(to, sample::MaskedKSpace)
    kspace = Adapt.adapt_structure(to, sample.kspace)
    mask = Adapt.adapt_structure(to, sample.mask)
    center_mask = Adapt.adapt_structure(to, sample.center_mask)
    sense_maps = Adapt.adapt_structure(to, sample.sense_maps)
    L = Adapt.adapt_structure(to, sample.L)
    MaskedKSpace(kspace, mask, center_mask, sense_maps, L, Sample.name)
end
function CUDA.unsafe_free!(s::MaskedKSpace)
    CUDA.unsafe_free!(s.kspace)
    CUDA.unsafe_free!(s.mask)
    CUDA.unsafe_free!(s.center_mask)
    CUDA.unsafe_free!(s.sense_maps)
    CUDA.unsafe_free!(s.L)
end
function MLUtils.batch(xs::Vector{<:MaskedKSpace})
    MaskedKSpace(
        MLUtils.batch([x.kspace for x in xs]), 
        MLUtils.batch([x.mask for x in xs]), 
        MLUtils.batch([x.center_mask for x in xs]), 
        MLUtils.batch([x.sense_maps for x in xs]), 
        MLUtils.batch([x.L for x in xs]), 
        MLUtils.batch([x.name for x in xs]),
    )
end

struct PairedMaskedKSpace{Tk, Tm, Tc, Ts, Tσ, Ti} <: ObservedKSpace 
    kspace::Tk
    mask::Tm
    center_mask::Tc
    sense_maps::Ts
    L::Tσ          # sqrt-covmat or noise-level map
    image::Ti
    name
end
input(s::PairedMaskedKSpace) = s.kspace
target(s::PairedMaskedKSpace) = s.kspace
reference(s::PairedMaskedKSpace) = s.image
noise_level(s::PairedMaskedKSpace{Tk, Tm, Tc, Ts, <:AbstractArray{T,4}}) where {Tk, Tm, Tc, Ts, T} = s.L
    
function Adapt.adapt_structure(to, sample::PairedMaskedKSpace)
    kspace = Adapt.adapt_structure(to, sample.kspace)
    mask = Adapt.adapt_structure(to, sample.mask)
    center_mask = Adapt.adapt_structure(to, sample.center_mask)
    sense_maps = Adapt.adapt_structure(to, sample.sense_maps)
    L = Adapt.adapt_structure(to, sample.L)
    image = Adapt.adapt_structure(to, sample.image)
    PairedMaskedKSpace(kspace, mask, center_mask, sense_maps, L, image, sample.name)
end
function CUDA.unsafe_free!(s::PairedMaskedKSpace)
    CUDA.unsafe_free!(s.kspace)
    CUDA.unsafe_free!(s.mask)
    CUDA.unsafe_free!(s.center_mask)
    CUDA.unsafe_free!(s.sense_maps)
    CUDA.unsafe_free!(s.L)
    CUDA.unsafe_free!(s.image)
end
function MLUtils.batch(xs::Vector{<:PairedMaskedKSpace})
    PairedMaskedKSpace(
        MLUtils.batch([x.kspace for x in xs]), 
        MLUtils.batch([x.mask for x in xs]), 
        MLUtils.batch([x.center_mask for x in xs]), 
        MLUtils.batch([x.sense_maps for x in xs]), 
        MLUtils.batch([x.L for x in xs]), 
        MLUtils.batch([x.image for x in xs]), 
        MLUtils.batch([x.name for x in xs]),
    )
end

struct TripleMaskedKSpace{Tk, Tm, Tc, Ts, Tσ, Ti} <: ObservedKSpace 
    kspace_a::Tk
    kspace_b::Tk
    mask::Tm
    center_mask::Tc
    sense_maps::Ts
    L::Tσ          # sqrt-covmat or noise-level map
    image::Ti
    name
end
kspace(s::TripleMaskedKSpace) = s.kspace_a
input(s::TripleMaskedKSpace) = s.kspace_a
target(s::TripleMaskedKSpace) = s.kspace_b
reference(s::TripleMaskedKSpace) = s.image
noise_level(s::TripleMaskedKSpace{Tk, Tm, Tc, Ts, <:AbstractArray{T,4}}) where {Tk, Tm, Tc, Ts, T} = s.L
    
function Adapt.adapt_structure(to, sample::TripleMaskedKSpace)
    kspace_a = Adapt.adapt_structure(to, sample.kspace_a)
    kspace_b = Adapt.adapt_structure(to, sample.kspace_b)
    mask = Adapt.adapt_structure(to, sample.mask)
    center_mask = Adapt.adapt_structure(to, sample.center_mask)
    sense_maps = Adapt.adapt_structure(to, sample.sense_maps)
    L = Adapt.adapt_structure(to, sample.L)
    image = Adapt.adapt_structure(to, sample.image)
    TripleMaskedKSpace(kspace_a, kspace_b, mask, center_mask, sense_maps, L, image, sample.name)
end
function CUDA.unsafe_free!(s::TripleMaskedKSpace)
    CUDA.unsafe_free!(s.kspace_a)
    CUDA.unsafe_free!(s.kspace_b)
    CUDA.unsafe_free!(s.mask)
    CUDA.unsafe_free!(s.center_mask)
    CUDA.unsafe_free!(s.sense_maps)
    CUDA.unsafe_free!(s.L)
    CUDA.unsafe_free!(s.image)
end
function MLUtils.batch(xs::Vector{<:TripleMaskedKSpace})
    TripleMaskedKSpace(
        MLUtils.batch([x.kspace_a for x in xs]), 
        MLUtils.batch([x.kspace_b for x in xs]), 
        MLUtils.batch([x.mask for x in xs]), 
        MLUtils.batch([x.center_mask for x in xs]), 
        MLUtils.batch([x.sense_maps for x in xs]), 
        MLUtils.batch([x.L for x in xs]), 
        MLUtils.batch([x.image for x in xs]), 
        MLUtils.batch([x.name for x in xs]),
    )
end

