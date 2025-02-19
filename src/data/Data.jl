module Data

using ..GroupCDL
using ..Samples
using ..Operators

using CUDA
import DataAugmentation as DA
using Distributions: DiscreteNonParametric
using MLUtils, MLDatasets
using Images
using Random
using HDF5
using DelimitedFiles
using Statistics
using Lux

export select_loaders

include("natural_image.jl")
include("modl.jl")

function select_loaders(; data_type="natural_image", kws...)
    t = replace(lowercase(data_type), "_"=>"")
    if t == "naturalimage"
        return get_image_dataloaders(; kws...)
    elseif t == "modl" || t == "modlbrain"
        return get_modlbrain_dataloaders(; kws...)
    else
        throw(ErrorException("loader-type $data_type not implemented."))
    end
end

end;
