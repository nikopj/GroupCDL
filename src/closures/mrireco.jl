#==============================================================================
                       ABSTRACT MRIReco CLOSURE
==============================================================================#

abstract type AbstractMRIRecoClosure <: AbstractClosure end

function (clo::AbstractMRIRecoClosure)(::Val{true}, sample::Samples.ObservedKSpace, net, ps, st)
    ref = reference(sample)
    y = input(sample)
    A = obs_operator(sample) 
    y_zf = A'(y)

    x̂, st = net((y, A), ps, st)

    smaps = A.map
    sense_mask = (sum(abs2, smaps; dims=3) .> 0)
    x̂ .*= sense_mask

    mv = 1f0 
    mse  = mean(abs2, abs.(ref) - abs.(x̂))
    psnr = 20log10(mv) - 10log10(mse)
    ssim = SSIMLoss.ssim(abs.(ref), abs.(x̂); peakval=mv)
    name = Samples.name(sample)
    name = name isa Vector ? name[1] : name

    sample_metrics = Dict(:name=>name, :psnr=>psnr, :ssim=>ssim, :mse=>mse, :maxval=>mv)
    outputs = Dict(:name      => name,
                   :input     => Array(abs.(y_zf)),
                   :reference => Array(abs.(ref)),
                   :output    => Array(abs.(x̂)),
                   :residual  => Array(abs.(abs.(ref) - abs.(x̂))),
                   :extrema   => extrema(abs, x̂),
                   :maxval    => mv,
                   :psnr      => psnr,
                   :ssim      => ssim,
                   :result    => (x̂, st),)
    return sample_metrics, outputs
end

metric_keys(::AbstractMRIRecoClosure) = (:psnr, :ssim, :mse, :maxval)
function init_metrics(closure::AbstractMRIRecoClosure, N) 
    Dict(:name=>Vector{String}(undef, N), 
         :psnr=>Vector{Float32}(undef, N),
         :mse=>Vector{Float32}(undef, N),
         :maxval=>Vector{Float32}(undef, N),
         :ssim=>Vector{Float32}(undef, N))
end

function log_outputs(::AbstractMRIRecoClosure, logger, outputs; logname=x->"val_image/$(basename(outputs[:name]))")
    imgs = clamp.(cat(outputs[:reference], outputs[:input], outputs[:output], 2outputs[:residual]; dims=4), 0, outputs[:maxval])
    imgs ./= outputs[:maxval]
    mos = cat([mosaicview(imgs[:,:,i,:]; nrow=2, rowmajor=true, npad=10) for i=1:size(imgs, 3)]...; dims=3) 
    log_image(logger, logname(outputs), mos, HWC)
end

"""
    SupervisedMRIReco(loss_type::String) <: AbstractcsmriClosure

Requires a paired kspace sample.
"""
struct SupervisedMRIReco <: AbstractMRIRecoClosure
    loss::Function
    function SupervisedMRIReco(; loss_type::String="mse")
        new(select_loss(loss_type))
    end
end

function Base.show(io::IO, clo::SupervisedMRIReco)
    print(io, "SupervisedMRIReco($(clo.loss))")
end

function (clo::SupervisedMRIReco)(::Val{false}, sample::Samples.ObservedKSpace, net, ps, st)
    x = reference(sample)
    y = input(sample)
    A = obs_operator(sample) 

    smaps = A.map
    sense_mask = (sum(abs2, smaps; dims=3) .> 0)

    lossval, grads = Zygote.withgradient(ps) do Θ
        x̂, st = net((y, A), Θ, st)
        clo.loss(x .* sense_mask, x̂ .* sense_mask)
    end

    return lossval, grads[1], st
end

#========================================
          Synthetic MRIReco
========================================#

struct SyntheticMRIReco <: AbstractSyntheticClosure
    fixed_rng::Bool
    rng_seed::Int
    baseclo::AbstractClosure
    accel::Int
    center_frac::Number
    uniform_offset::Union{Missing, Int}
    mask_type::Union{String, Symbol}
    readout_dir::Union{String, Symbol}
    noise_level::Union{Vector, Tuple}
    noise_jitter::Float32
    max_corr::Float32
    function SyntheticMRIReco(; 
            fixed_rng = false,
            rng_seed = 0,
            baseclo_type="supervisedmrireco", 
            accel=4, 
            center_frac=0.08, 
            mask_type=:random, 
            readout_dir=:horizontal, 
            noise_level=(0f0, 0f0), 
            noise_jitter=0f0, 
            max_corr=0f0, 
            uniform_offset=missing,
            baseclo_kws...)

        if isnothing(uniform_offset)
            uniform_offset = missing
        end

        readout_dir = if readout_dir in (:horizontal, "horizontal", "h", :h)
            :h
        elseif readout_dir in (:vertical, "vertical", "v", :v)
            :v
        else 
            throw(ErrorException("readout_dir $readout_dir not implemented."))
        end

        baseclo = select_closure(; closure_type=baseclo_type, baseclo_kws...)

        new(fixed_rng, rng_seed, baseclo, accel, center_frac, uniform_offset, mask_type, readout_dir,
            Float32.(noise_level), Float32(noise_jitter), Float32(max_corr))
    end
end

function Base.show(io::IO, clo::SyntheticMRIReco) 
    print(io, 
"""
SyntheticMRIReco(baseclo=$(clo.baseclo), fixed_rng=$(clo.fixed_rng), rng_seed=$(clo.rng_seed),
                 mask=$(clo.mask_type), readout_dir=$(clo.readout_dir), offset=$(clo.uniform_offset),
                 accel=$(clo.accel), center_frac=$(clo.center_frac), 
                 noise_level=$(clo.noise_level), noise_jitter=$(clo.noise_jitter), max_corr=$(clo.max_corr))
""")
end

"""
    genobs(clo::SyntheticMRIReco, sample, noise_level)

must return a ObservedKSpace sample
"""
function genobs(clo::SyntheticMRIReco, sample::Samples.GTMoDLImage, noise_level, vol_mask=false, seed=missing)
    if !ismissing(seed) || clo.fixed_rng
        # reasoning: volumes all have the same mask (vol_rng)
        # slices have different noise realizations (slice_rng), but the same noise-level (vol_rng)
        # this all assumes a batch-size of 1
        name = Samples.name(sample)
        name = (name isa Vector) ? name[1] : name
        slicename = basename(name)
        idx = findfirst("_slice", slicename)
        volname = slicename[1:idx[1]-1]

        vol_seed = ismissing(seed) ? (hash(volname) + clo.rng_seed) : seed
        slice_seed = ismissing(seed) ? (hash(slicename) + clo.rng_seed) : seed

        vol_rng = MersenneTwister(vol_seed)
        slice_rng = MersenneTwister(slice_seed)
    else
        vol_rng = Random.TaskLocalRNG()
        slice_rng = Random.TaskLocalRNG()
    end

    x = sample.image
    xmc = Sense(sample.smaps)(x)    # multicoil ground-truth image
    σd = dropdims(Operators.rand_range(vol_rng, x, noise_level); dims=1)

    L = Operators.rand_sqrtcov(vol_rng, xmc, σd, clo.noise_jitter, clo.max_corr)
    y = acgn(slice_rng, xmc, L)

    N = clo.readout_dir == :h ? size(x, 1) : size(x,2)
    B = vol_mask ? 1 : size(x, 4)

    am, cm = GroupCDL.generate_mask(N, B; type=clo.mask_type, accel=clo.accel, center_frac=clo.center_frac, return_separate_mask=true, readout_dir=clo.readout_dir, offset=clo.uniform_offset, rng=vol_rng) .|> cu
    m = (am .|| cm)
    M = Mask(m, cm)
    F = Fourier{2}()

    k = M(F(y))

    return Samples.PairedMaskedKSpace(k, M.mask, M.center_mask, sample.smaps, L, x, Samples.name(sample))
end
genobs(clo::SyntheticMRIReco, v::Val{false}, sample) = genobs(clo, sample, clo.noise_level, false)
genobs(clo::SyntheticMRIReco, v::Val{true}, sample) = genobs(clo, sample, mean(clo.noise_level), true)
