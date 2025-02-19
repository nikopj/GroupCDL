module Samples

import Adapt
using MLUtils
using CUDA

using ..Operators
using ..GroupCDL: mul_channel, rss
import Adapt
using MLUtils
using CUDA

export AbstractSample, Image, NoisyImage
export input, target, reference, maxval, name, noise_level
export obs_operator

#==============================================================================
                            STANDARD SAMPLES
==============================================================================#

abstract type AbstractSample end
abstract type Image <: AbstractSample end
abstract type NoisyImage <: Image end

image(s::Image) = input(s)
input(::Image) = missing
target(::Image) = missing
reference(::Image) = missing
maxval(::Image) = missing
name(::Image) = missing
noise_level(::Image) = missing

include("natural_image.jl")
include("modl.jl")
include("kspace.jl")

end;
