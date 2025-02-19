struct Collect <: DA.Transform end

function DA.apply(::Collect, item::DA.ArrayItem; randstate = nothing)
    return DA.ArrayItem(collect(DA.itemdata(item)))
end

struct Show <: DA.Transform end

function DA.apply(::Show, item; randstate = nothing)
    @show typeof(item)
    return item
end

ImageTestTransform() = DA.compose(
    DA.ImageToTensor(),
)
ImageTrainTransform(cs::Integer) = DA.compose(
    DA.Maybe(DA.FlipX(), 0.5),
    DA.Maybe(DA.FlipY(), 0.5),
    DA.Rotate(DiscreteNonParametric([0,90,180,270], 0.25*ones(4))),
    DA.RandomCrop((cs, cs)), 
    DA.ImageToTensor(), 
    DA.ToEltype(Float32), 
    Collect(),
)

function load_image(path::AbstractString, tfm=DA.ImageToTensor(); gray=false)
    tfm = if gray
        DA.compose(DA.ToEltype(Gray), tfm) 
    else
        DA.compose(DA.ToEltype(RGB), tfm) 
    end

    item = DA.Image(load(path))
    item = DA.apply(tfm, item)
    return DA.itemdata(item) 
end

struct ImageDataset{online} <: MLDatasets.AbstractDataContainer
    transform::DA.Transform
    images::Union{Tuple, Vector}
    paths::Vector{String}
end

function Base.getindex(ds::ImageDataset{false}, i::Integer) 
    item = DA.Image(ds.images[i])
    item = DA.apply(ds.transform, item)
    Samples.GTNaturalImage(DA.itemdata(item), ds.paths[i])
end

function Base.getindex(ds::ImageDataset{true}, i::Integer) 
    item = DA.Image(load(ds.paths[i]))
    item = DA.apply(ds.transform, item)
    Samples.GTNaturalImage(DA.itemdata(item), ds.paths[i])
end

Base.getindex(ds::ImageDataset, is::AbstractVector) = map(Base.Fix1(getobs, ds), is)
Base.length(ds::ImageDataset) = length(ds.paths)

ImageDataset(transform, paths::Vector{<:AbstractString}) = ImageDataset(transform, load.(paths), paths)
ImageDataset(transform, ds::MLDatasets.FileDataset) = ImageDataset(transform, ntuple(i->ds[i], length(ds)), ds.paths)

function ImageDataset(paths::Vector{<:AbstractString}; online=false, cropsize=nothing, grayscale=false)
    # setup transform
    tfm = if isnothing(cropsize) || ismissing(cropsize)
        ImageTestTransform()
    else
        ImageTrainTransform(cropsize)
    end

    tfm = if grayscale 
        DA.compose(DA.ToEltype(Gray), tfm) 
    else
        DA.compose(DA.ToEltype(RGB), tfm) 
    end

    # get filepaths
    paths = reduce(vcat, [(isdir(p) ? readdir(p; join=true) : p) for p in paths])
    if paths isa String
        paths = [paths,]
    end

    # get dataset
    ds = if online
        # FileDataset(Base.Fix2(load_image, tfm), paths)
        ImageDataset{true}(tfm, (nothing,), paths)
    else
        ImageDataset{false}(tfm, load.(paths), paths)
    end

    return ds
end

# collate=true: batches items into single array
# partial=false: drops last mini-batch if not up to batchsize
function get_image_dataloaders(;
        trainpaths::Vector = missing, 
        valpaths::Vector   = missing, 
        testpaths::Vector  = missing,
        cropsize  = 128, 
        batchsize = 10, 
        grayscale = false,
        parallel = false,
        buffer = false,
        online = false,
        rng = Random.GLOBAL_RNG,
        total_workers = 1,
        backend = missing,
    )

    ds_train = ImageDataset(trainpaths; cropsize=cropsize, grayscale=grayscale, online=online)
    ds_val = ImageDataset(valpaths; cropsize=nothing, grayscale=grayscale, online=online)
    ds_test = ImageDataset(testpaths; cropsize=nothing, grayscale=grayscale, online=online)

    if !ismissing(backend) && (total_workers > 1)
        ds_train = DistributedUtils.DistributedDataContainer(backend, ds_train)
    end

    dl_train = DataLoader(ds_train;
        batchsize = batchsize ÷ total_workers,
        collate   = true, 
        partial   = true,
        buffer    = buffer,
        parallel  = parallel,
        shuffle   = true,
        rng = rng,
    )

    dl_val = DataLoader(ds_val;
        batchsize = 1, 
        collate   = true, 
        partial   = true,
        buffer    = false,
        parallel  = false,
        shuffle   = false,
    )

    dl_test = DataLoader(ds_test;
        batchsize = 1, 
        collate   = true, 
        partial   = true,
        buffer    = false,
        parallel  = false,
        shuffle   = false,
    )
    return (train=dl_train, val=dl_val, test=dl_test)
end
