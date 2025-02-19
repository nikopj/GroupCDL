module GroupCDL

using Base.GC
using Random, Statistics, LinearAlgebra, FFTW
using Accessors

using CUDA, cuDNN
using NNlib, Lux, LuxCUDA
import LuxCore
import MPI, NCCL
using MLUtils
using Functors

using Optimisers, Zygote, ParameterSchedulers
import ChainRulesCore as CRC
using SSIMLoss
using CirculantAttention
using LazyGrids

using FileIO, Images, HDF5, BSON, NPZ
using DelimitedFiles, TensorBoardLogger, Logging
using MosaicViews

using Printf
import ProgressMeter as PM

include("operators.jl")
using .Operators
export AWGN, AddNoise, Sense, Fourier, Mask
export awgn, acgn, gramian

include("utils.jl")
include("nle.jl")
include("mask.jl")
include("solver.jl")

include("samples/Samples.jl")
using .Samples

include("data/Data.jl")
using .Data
export select_loaders

include("networks/layers.jl")
include("networks/gabor.jl")
include("networks/lista.jl")
include("networks/group.jl")
include("networks/preproc.jl")
include("networks/networks.jl")
export select_network, GroupCDLNet, CDLNet

include("closures/Closures.jl")
using .Closures
export select_closure

include("train.jl")
export eval_epoch, train_epoch!, train!

end;
