using GroupCDL
using ProgressMeter
using Statistics
using Lux
using MLUtils
using Printf
using HDF5

include("../main.jl")
device = Lux.gpu_device()

# fn = "/gpfs/scratch/npj226/GroupCDL/trained_nets/CDL-modl8-crop-2_2205/config.yaml"
# fn = "/gpfs/scratch/npj226/GroupCDL/trained_nets/CDL-modl4-1_2208/config.yaml"

savedir = joinpath(dirname(fn), "results")
mkpath(savedir)

DUMMY = false
NOTE = "mag"

logfn = joinpath(savedir, (DUMMY ? "log_zf.txt" : "log.txt"))
savefn = joinpath(savedir, (DUMMY ? "zf.h5" : "eval.h5"))

if !isfile(logfn)
    open(logfn, "a") do fid
        write(fid, "PSNR SSIM NOTE\n")
    end
end

net, ps, st, ot = main(; config=fn, warmup=false, verbose=true, alt_ckpt_fn="net_psnr.bson", get_loaders=true)

if DUMMY
    net, ps, st = GroupCDL.DummyNetwork(), NamedTuple(), NamedTuple()
end

clo = ot.closure
dl = ot.loaders.test

denoisedv = []
psnrv = zeros(length(dl))
ssimv = zeros(length(dl))

for (ii, sample) in enumerate(dl)
    print("$ii\r")
    sample = sample |> device
    _, output = clo(Val(true), sample, net, ps, st)

    push!(denoisedv, output[:result][1])
    psnrv[ii] = output[:psnr]
    ssimv[ii] = output[:ssim]
end

xhat = Array(cat(denoisedv...; dims=4)[:,:,1,:])

h5open(savefn, "w") do fid
    fid["data"] = xhat
    fid["psnrv"] = psnrv
    fid["ssimv"] = ssimv
end

avg_psnr = mean(psnrv)
avg_ssim = mean(ssimv)

open(logfn, "a") do fid
    write(fid, @sprintf("%.2f %.2f %s\n", avg_psnr, 100avg_ssim, NOTE))
end
@printf("%.2f %.2f %s\n", avg_psnr, 100avg_ssim, NOTE)
