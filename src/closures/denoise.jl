#==============================================================================
                      ABSTRACT DENOISE CLOSURE
==============================================================================#

abstract type AbstractDenoiseClosure <: AbstractClosure end

function (clo::AbstractDenoiseClosure)(::Val{true}, sample::NoisyImage, net, ps, st)
    y = input(sample)
    A = AWGN(noise_level(sample))

    x̂, st = net((y, A), ps, st)

    mv   = maxval(sample)

    ref  = reference(sample)
    mse  = mean(abs2, ref - x̂)
    psnr = 20log10(mv) - 10log10(mse)
    ssim = SSIMLoss.ssim(abs.(ref), abs.(x̂); peakval=mv)

    sample_metrics = Dict(:name=>Samples.name(sample), :psnr=>psnr, :ssim=>ssim, :mse=>mse, :maxval=>mv)
    outputs = Dict(:name      => Samples.name(sample),
                   :input     => Array(abs.(y)),
                   :reference => Array(abs.(ref)),
                   :output    => Array(abs.(x̂)),
                   :residual  => Array(abs.(ref - x̂)),
                   :extrema   => extrema(abs, x̂),
                   :psnr      => psnr,
                   :ssim      => ssim,
                   :result    => (x̂, st),)
    return sample_metrics, outputs
end

metric_keys(::AbstractDenoiseClosure) = (:psnr, :ssim, :mse, :maxval)
function init_metrics(closure::AbstractDenoiseClosure, N)
    Dict(:name=>Vector{String}(undef, N),
         :psnr=>Vector{Float32}(undef, N),
         :mse=>Vector{Float32}(undef, N),
         :maxval=>Vector{Float32}(undef, N),
         :ssim=>Vector{Float32}(undef, N))
end

function log_outputs(::AbstractDenoiseClosure, logger, outputs; logname=x->"val_image/$(basename(outputs[:name]))")
    imgs = clamp.(cat(outputs[:reference], outputs[:input], outputs[:output], 2outputs[:residual]; dims=4), 0, 1)
    mos = cat([mosaicview(imgs[:,:,i,:]; nrow=2, rowmajor=true, npad=10) for i=1:size(imgs, 3)]...; dims=3)
    log_image(logger, logname(outputs), mos, HWC)
end

"""
    SupervisedDenoise(loss_type::String) <: AbstractDenoiseClosure

Requires a paired image sample.
"""
struct SupervisedDenoise <: AbstractDenoiseClosure
    loss::Function
    function SupervisedDenoise(; loss_type::String="mse")
        new(select_loss(loss_type))
    end
end

function Base.show(io::IO, clo::SupervisedDenoise)
    print(io, "SupervisedDenoise($(clo.loss))")
end

# Lux.Training-compatible loss function: (model, ps, st, data) -> (loss, st, extras)
function (clo::SupervisedDenoise)(net, ps, st, sample::NoisyImage)
    x = target(sample)
    y = input(sample)
    A = AWGN(noise_level(sample))
    x̂, st = net((y, A), ps, st)
    return clo.loss(x, x̂), st, (;)
end

#==============================================================================
                           SYNTHETIC CLOSURES
==============================================================================#

"""
    AbstractSyntheticClosure

A wrapper around a base-closure. Takes in unpaired (gt) sample, generates
paired data by implementing `syn_sample = genobs(synclo, v::Val, sample)`, and
passes the synthetic sample to the base closure.
"""
abstract type AbstractSyntheticClosure <: AbstractClosure end
metric_keys(clo::AbstractSyntheticClosure) = metric_keys(clo.baseclo)
init_metrics(clo::AbstractSyntheticClosure, N) = init_metrics(clo.baseclo, N)
log_outputs(clo::AbstractSyntheticClosure, logger, outputs) = log_outputs(clo.baseclo, logger, outputs)

# Eval path: generate synthetic sample then delegate to baseclo
function (synclo::AbstractSyntheticClosure)(v::Val{true}, sample, args...)
    syn_sample = genobs(synclo, v, sample)
    synclo.baseclo(v, syn_sample, args...)
end

# Lux.Training-compatible training path: generate synthetic sample then delegate
function (synclo::AbstractSyntheticClosure)(net, ps, st, sample)
    syn_sample = genobs(synclo, Val(false), sample)
    synclo.baseclo(net, ps, st, syn_sample)
end

struct SyntheticAWGN <: AbstractSyntheticClosure
    baseclo::AbstractClosure
    noise_level::Union{Tuple, Vector}
    noisy_target::Bool
    function SyntheticAWGN(;
            noise_level = (25, 25),
            noisy_target = false,
            maxval = 255,
            baseclo_type::String="supervised_denoise", baseclo_kws...)
        new(select_closure(; closure_type=baseclo_type, baseclo_kws...),
            Float32.(noise_level ./ maxval),
            noisy_target)
    end
end

function Base.show(io::IO, clo::SyntheticAWGN)
    print(io,
"""
SyntheticAWGN(
    baseclo=$(clo.baseclo),
    noise_level=$(clo.noise_level),
    noisy_target=$(clo.noisy_target)
)
"""
    )
end

function genobs(clo::SyntheticAWGN, sample::Image, noise_level)
    x = Samples.image(sample)
    σ = Operators.rand_range(x, noise_level)
    y = awgn(x, σ)

    new_sample = if clo.noisy_target
        z = awgn(x, σ)
        Samples.TripleNaturalImage(x, y, z, σ, sample.path)
    else
        Samples.PairedNaturalImage(x, y, σ, sample.path)
    end

    return new_sample
end
genobs(clo::SyntheticAWGN, v::Val{false}, sample::Image) = genobs(clo, sample, clo.noise_level)
genobs(clo::SyntheticAWGN, v::Val{true}, sample::Image) = genobs(clo, sample, mean(clo.noise_level))
