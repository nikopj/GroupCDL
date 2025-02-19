module Closures

using ..GroupCDL
using ..GroupCDL: totuple
using ..Operators # for synthetic closures
using ..Samples

using Random
using CUDA
using NNlib
using Zygote
using Statistics
using SSIMLoss
using MLUtils
using MosaicViews
using Logging, TensorBoardLogger
include("loss.jl") # get select_loss

export AbstractClosure, select_closure
export log_outputs, metric_keys, init_metrics, reduce_metrics

abstract type AbstractClosure end

(clo::AbstractClosure)(b::Bool, args...) = clo(Val(b), args...)

metric_keys(::AbstractClosure) = missing
init_metrics(::AbstractClosure) = missing
log_outputs(::AbstractClosure, logger, outputs) = missing

include("denoise.jl")
include("mrireco.jl")

function select_closure(; closure_type::String="syntheticawgn", kws...)
    t = replace(lowercase(closure_type), "_"=>"")
    clo = if t == "superviseddenoise"
        SupervisedDenoise(; kws...)
    elseif t == "syntheticawgn" 
        SyntheticAWGN(; kws...)
    elseif t == "supervisedmrireco"
        SupervisedMRIReco(; kws...)
    elseif t == "syntheticmrireco"
        SyntheticMRIReco(; kws...)
    else
        throw(ErrorException("Closure \"$closure_type\" not implemented."))
    end
    return clo
end

# this function combines all metrics which correspond to the same name name.
# useful for computing volume statistics when each metric entry represents
# a single slice metric
function reduce_metrics(clo::AbstractClosure, metrics::Dict)
    names = unique(metrics[:name])
    uniquemet = Dict(:name=>names,  [k=>Vector{Union{Vector, Missing}}(missing, length(names)) for k in metric_keys(clo)]...)

    # go through each element of metrics
    for (i, p) in enumerate(metrics[:name])
        # find corresponding name-index in reduced metrics dict (uniquemet)
        j = findfirst(x->x==p, names)

        # for each metric
        for k in metric_keys(clo)

            # push the metric onto the list for this unique name (or start the list)
            if ismissing(uniquemet[k][j])
                uniquemet[k][j] = [metrics[k][i],]
            else
                push!(uniquemet[k][j], metrics[k][i])
            end
        end
    end

    redmet = Dict(:name=>names,  [k=>Vector{Float32}(undef, length(names)) for k in metric_keys(clo)]...)

    # compute statistics necessary for psnr, nmse, etc.
    for i in 1:length(redmet[:name])
        for k in metric_keys(clo)
            if k == :maxval
                redmet[k][i] = maximum(uniquemet[k][i])
            elseif k == :normval
                redmet[k][i] = sqrt(sum(abs2, uniquemet[k][i]))
            elseif k == :mse
                redmet[k][i] = mean(uniquemet[k][i])
            end
        end
    end

    # reduce the list for each metric
    for i in 1:length(redmet[:name])
        for k in metric_keys(clo)
            if k == :psnr
                redmet[k][i] = 20log10(redmet[:maxval][i]) - 10log10(redmet[:mse][i])
            elseif k == :nmse
                redmet[k][i] = redmet[:mse][i] / (redmet[:normval][i]^2)
            elseif k == :ssim
                redmet[k][i] = mean(uniquemet[k][i])
            end
        end
    end

    return redmet
end

end;

