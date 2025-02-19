MoDL_TrainTransform(cs::Integer) = begin
    function tfm(xs::Tuple) 
        if rand() > 0.5
            xs = [selectdim(x, 1, size(x,1):-1:1) for x in xs]
        end
        if rand() > 0.5
            xs = [selectdim(x, 2, size(x,2):-1:1) for x in xs]
        end
        M, N = size(xs[1])[1:2]
        a, b = rand(1:M-cs+1), rand(1:N-cs+1)
        xs = [selectdim(x, 1, a:a+cs-1) for x in xs]
        xs = [selectdim(x, 2, b:b+cs-1) for x in xs]
        return xs
    end
end

MoDL_TransposeTransform() = begin
    function tfm(xs::Tuple)
        [permutedims(x, (2,1,[n for n=3:ndims(x)]...)) for x in xs]
    end
end

struct MoDLBrainDataset{online,test,noisy} <: MLDatasets.AbstractDataContainer
    filename::String
    transform
    coilcombined::Union{Missing, AbstractArray}
    sensemaps::Union{Missing, AbstractArray}
    mask::Union{Missing, AbstractArray}
    slicenums::Vector{Int}
    noisymc::Union{Missing, AbstractArray}
    noisycov::Union{Missing, AbstractArray}

    function MoDLBrainDataset(; 
            fn::String="dataset/MODL_BRAIN.hdf5", 
            slicenums::Union{Colon, Vector{Int}}=:,
            cropsize = missing,
            online = false,
            ongpu = false,
            test = false,
            transpose = false,
            noisy_filename = missing,
        )
        prefix = test ? "tst" : "trn"

        if transpose
            transform = MoDL_TransposeTransform()
            @warn "Using MoDL_TransposeTransform"
        else
            transform = (test || isnothing(cropsize) || ismissing(cropsize) || cropsize==0) ? identity : MoDL_TrainTransform(cropsize)
        end

        coilcombined = online ? missing : h5read(fn, prefix*"Org")
        sensemaps = online ? missing : h5read(fn, prefix*"Csm")
        mask = online ? missing : h5read(fn, prefix*"Mask")

        if !ismissing(noisy_filename)
            @assert !online
            noisymc = h5read(noisy_filename, prefix*"_data")
            noisycov = h5read(noisy_filename, prefix*"_cov")
            if ongpu
                noisymc  = noisymc |> cu
                noisycov = noisycov |> cu
            end
        else
            noisymc = missing
            noisycov = missing
        end

        if ongpu
            coilcombined = coilcombined |> cu
            sensemaps = sensemaps |> cu
            mask = mask |> cu
        end

        if slicenums isa Colon
            h5open(fn, "r") do fid
                slicenums = collect(1:size(fid[prefix*"Org"], 3))
            end
        end

        noisy = !ismissing(noisy_filename)

        new{online, test, noisy}(fn, transform, coilcombined, sensemaps, mask, slicenums, noisymc, noisycov)
    end
end

function Base.getindex(ds::MoDLBrainDataset{false,test,false}, idx::Integer) where test
    i = ds.slicenums[idx]
    x = ds.coilcombined[:, :, i:i] 
    s = ds.sensemaps[:, :, :, i] 
    x, s = ds.transform((x, s))
    Samples.GTMoDLImage(x, s, i)
end

function Base.getindex(ds::MoDLBrainDataset{true,test,false}, idx::Integer) where test
    local x, s, m
    i = ds.slicenums[idx]
    prefix = test ? "tst" : "trn"
    h5open(ds.filename, "r") do fid
        x = fid[prefix*"Org"][:,:,i:i] 
        s = fid[prefix*"Csm"][:,:,:,i] 
    end
    data = ds.transform((x, s))
    Samples.GTMoDLImage(x, s, i)
end

function Base.getindex(ds::MoDLBrainDataset{false,test,true}, idx::Integer) where test
    i = ds.slicenums[idx]
    x = ds.coilcombined[:, :, i:i] 
    s = ds.sensemaps[:, :, :, i] 
    ymc = ds.noisymc[:, :, :, i, :] 
    cov = ds.noisycov[:, :, i] 
    x, s, ymc = ds.transform((x, s, ymc))
    Samples.PairedMoDLImage(x, ymc, s, cov, i)
end

Base.getindex(ds::MoDLBrainDataset, is::AbstractVector) = map(Base.Fix1(getobs, ds), is)
Base.length(ds::MoDLBrainDataset) = length(ds.slicenums)

function get_modlbrain_dataloaders(;
        fn = "dataset/MODL_BRAIN.hdf5",
        cropsize  = 0, 
        batchsize = 2, 
        parallel = false,
        buffer = false,
        split = 0.95,
        online = false,
        ongpu = false,
        rng = Random.GLOBAL_RNG,
        transpose = false,
        noisy_filename = missing, #"compare/modl_noisy2datamc_20_001_02GRAPPA.h5",
        total_workers = 1,
        backend = missing,
    )
    local nslices

    if !ismissing(noisy_filename)
        @assert ongpu "Use of noisy MODL data requires ongpu=true"
    end

    h5open(fn, "r") do fid
        nslices = size(fid["trnOrg"], 3)
    end

    slicenums = shuffle(collect(1:nslices))
    N = round(Int, split * nslices)
    trn_slices, val_slices = slicenums[1:N], slicenums[N+1:end]

    ds_train = MoDLBrainDataset(fn=fn, slicenums=trn_slices, cropsize=cropsize, noisy_filename=noisy_filename, online=online, test=false, ongpu=ongpu)
    ds_test  = MoDLBrainDataset(fn=fn, online=online, test=true, noisy_filename=noisy_filename, transpose=transpose)

    if length(val_slices) > 0
        ds_val = MoDLBrainDataset(fn=fn, slicenums=val_slices, online=online, noisy_filename=noisy_filename, test=false, transpose=transpose)
    else
        ds_val = ds_test
    end

    dl_train = DataLoader(
        ds_train;
        batchsize = batchsize ÷ total_workers,
        collate   = true, 
        partial   = false,
        buffer    = buffer,
        parallel  = parallel,
        shuffle   = true,
    )
    dl_val = DataLoader(
        ds_val;
        batchsize = 1, 
        collate   = true, 
        partial   = true,
        buffer    = false,
        parallel  = false,
        shuffle   = false,
    )
    dl_test = DataLoader(
        ds_test;
        batchsize = 1, 
        collate   = true, 
        partial   = true,
        buffer    = false,
        parallel  = false,
        shuffle   = false,
    )
    return (train=dl_train, val=dl_val, test=dl_test)
end
