using GroupCDL
using cuDNN
using CUDA
using LuxCUDA
using Lux
import MPI, NCCL
using Random, Statistics
using ArgParse
using FileIO, BSON
using Accessors
using YAML, TensorBoardLogger, Logging

if LuxCUDA.functional()
    DistributedUtils.initialize(NCCLBackend)
    backend = DistributedUtils.get_distributed_backend(NCCLBackend)
else
    DistributedUtils.initialize(MPIBackend)
    backend = DistributedUtils.get_distributed_backend(MPIBackend)
end

const local_rank = DistributedUtils.local_rank(backend)
const total_workers = DistributedUtils.total_workers(backend)
is_distributed() = total_workers > 1
should_log() = !is_distributed() || (local_rank == 0)

CUDA.allowscalar(false)
include("scripts/makeconfig.jl")

function main(;
        config = missing,
        network_config = "config/cdlnet.yaml", 
        train_config = "config/train.yaml", 
        closure_config = "config/synthawgn_closure.yaml",
        data_config = "config/image_data.yaml",
        log_config = "config/log.yaml",
        seed = rand(1:9999), 
        device = gpu_device(), 
        train = false, 
        train_continue = false,
        warmup = false, 
        warmup_trn = true, 
        warmup_val = true, 
        verbose = false,
        eval = false,
        eval_closure_config = missing,
        eval_data_config = missing,
        eval_dummy = false,
        eval_noise_level = missing,
        mpi = false, # dummy argument
        get_loaders = true,
        worker_str = is_distributed() ? "[$(local_rank+1)/$(total_workers)]" : "",
        alt_ckpt_fn = missing,
    )
    config = loadconfig(config; 
                        network = network_config, 
                        train = train_config, 
                        closure = closure_config, 
                        data = data_config,
                        log = log_config)

    # initialize
    network = select_network(; config[:network]...)
    should_log() && verbose && (@info "Network:"; display(network))

    clo = select_closure(; config[:closure]...)
    should_log() && verbose && (@info "Closure:"; display(clo))

    if get_loaders
        loaders = select_loaders(; backend=backend, total_workers=total_workers, config[:data]...)
    else
        loaders = missing
    end

    if verbose && get_loaders
        @info("Train Loader $worker_str:"); display(loaders.train)
        @info("Val Loader $worker_str:"); display(loaders.val)
    end

    # set random seed
    rng = Random.default_rng()
    local_seed = is_distributed() ? seed + local_rank : seed
    Random.seed!(rng, local_seed)
    verbose && println("random seed set to $worker_str: $local_seed")

    start_epoch = 0
    backtrack_count = 0
    bestloss = Inf32
    ckpt_epoch = 0

    if config[:log][:ckpt] != nothing
        ckptfn = config[:log][:ckpt]
        ckptfn = ismissing(alt_ckpt_fn) ? ckptfn : joinpath(dirname(ckptfn), alt_ckpt_fn)
        ckpt = load(ckptfn)
        ps = ckpt[:ps] |> device

        if :st in keys(ckpt) || "st" in keys(ckpt)
            st = ckpt[:st] |> device
        else
            @warn "\"st\" not provided by checkpoint"
        end

        if :backtrack_count in keys(ckpt) || "backtrack_count" in keys(ckpt)
            backtrack_count = ckpt[:backtrack_count]
        else
            verbose && @warn "\"backtrack_count\" not provided by checkpoint"
        end

        if :bestloss in keys(ckpt) || "bestloss" in keys(ckpt)
            bestloss = ckpt[:bestloss]
        else
            verbose && @warn "\"bestloss\" not provided by checkpoint"
        end

        ckpt_epoch = ckpt[:epoch]
        verbose && println("Loading checkpoint $(ckptfn): epoch=$(ckpt[:epoch])")
        if train_continue
            start_epoch = ckpt[:epoch] 
        end
    else
        verbose && @info("Setting up network $worker_str...")
        ps, st = Lux.setup(rng, network) |> device
        verbose && println("done $(worker_str).")
    end

    if config[:log][:pretrain_config] != nothing
        pre_config = loadconfig(config[:log][:pretrain_config])
        pre_ckpt = load(pre_config[:log][:ckpt])
        pre_net = select_network(; pre_config[:network]...)
        pre_ps = pre_ckpt[:ps] |> device
        verbose && (println("Loading pretrained network $(config[:log][:pretrain_config]): epoch=$(pre_ckpt[:epoch]) $worker_str"); display(pre_net))
        ps = preload(network, ps, pre_net, pre_ps)
    end

    if is_distributed()
        ps = DistributedUtils.synchronize!!(backend, ps)
        st = DistributedUtils.synchronize!!(backend, st)
    end

    if :share in keys(config)
        if network isa GroupCDLNet
            ps = GroupCDL.share_parameters(network, ps, config[:share])

            K = GroupCDL.iters(network)
            fl = Lux.parameterlength(network.dict)
            num_params = (K * 2 * fl +       # AB 
                          fl +               # D
                          Lux.parameterlength(network.lista.layer.prox.nlss) +   # θϕρ
                          2 * Lux.parameterlength(network.lista.layer.prox.Wα) + # αβ
                          Lux.parameterlength(network.lista.layer.prox.γ) +      # γ
                          K * Lux.parameterlength(network.lista.layer.prox.τ))   # τ

            println("Approximate Num. Learned Params (M) $(worker_str): ", num_params / 1e6)
        end
    end

    if warmup
        if warmup_trn 
            verbose && @info "Starting train warmup $(worker_str)..."
            CUDA.@time sample = first(loaders.train) |> device
            CUDA.@time trn_warmup = clo(Val(false), sample, network, ps, st)
            _, _, st = trn_warmup
            verbose && should_log() && println("done.")
        else
            trn_warmup = missing
        end
        if warmup_val
            verbose && @info "Starting val warmup $(worker_str)..."
            CUDA.@time sample = first(loaders.val) |> device
            CUDA.@time val_warmup = clo(Val(true), sample, network, ps, Lux.testmode(st))
            verbose && should_log() && println("done.")
        else
            val_warmup = missing
        end
        out_warmup = (trn=trn_warmup, val=val_warmup)
    else
        out_warmup = (trn=missing, val=missing)
    end

    if train
        logdir0 = train_continue ? config[:log][:logdir] : config[:log][:logdir]*"_$seed"

        if should_log()
            logger = TBLogger(logdir0, train_continue ? tb_append : tb_increment)
            config[:log][:logdir] = logger.logdir
            config[:log][:ckpt] = joinpath(logger.logdir, "net.bson")
            saveyaml(joinpath(logger.logdir, "config.yaml"), config)
            verbose && @info "Saving config file to \"$(config[:log][:logdir])\""

            with_logger(logger) do 
                @info "config" network=config[:network] train=config[:train] closure=config[:closure] data=config[:data] log=config[:log]
            end
            logdir = logger.logdir
        else
            logger = missing
            logdir = logdir0
        end

        ps, st = train!(clo, loaders, network, ps, st; 
                        rng = rng,
                        device = device, 
                        logger = logger, 
                        logdir = logdir,
                        start  = start_epoch,
                        backtrack_count = backtrack_count,
                        bestloss = bestloss,
                        backend = backend,
                        should_log = should_log,
                        is_distributed = is_distributed,
                        local_rank = local_rank,
                        total_workers = total_workers,
                        config = config, # for saving inside checkpoint
                        config[:train]...)
    end

    if eval
        if !ismissing(eval_closure_config)
            eval_clo_config = loadyaml(eval_closure_config)[:closure]
            if !ismissing(eval_noise_level)
                eval_clo_config[:noise_level] = [eval_noise_level, eval_noise_level]
            end
        end

        eval_clo = ismissing(eval_closure_config) ? clo : select_closure(; eval_clo_config...)
        should_log() && verbose && (@info "Eval-Closure:"; display(eval_clo))

        dl = if !ismissing(eval_data_config)
            select_loaders(; loadyaml(eval_data_config)[:data]...).test
        else
            loaders.test
        end

        metrics = if eval_dummy
            eval_epoch(eval_clo, dl, GroupCDL.DummyNetwork(), NamedTuple(), NamedTuple(); desc="EVAL", verbose=verbose, device=device)
        else
            eval_epoch(eval_clo, dl, network, ps, st; desc="EVAL", verbose=verbose, device=device)
        end

        verbose && println([(k, mean(metrics[k])) for k in GroupCDL.metric_keys(eval_clo)])
        verbose && println([(k, std(metrics[k])) for k in GroupCDL.metric_keys(eval_clo)])
        red_metrics = GroupCDL.reduce_metrics(eval_clo, metrics)
    else
        eval_clo = missing
        metrics = missing
        red_metrics = missing
    end

    return network, ps, st, (loaders=loaders, 
                             config=config, 
                             closure=clo, 
                             eval_closure=eval_clo, 
                             reduced_metrics=red_metrics, 
                             metrics=metrics, 
                             warmup=out_warmup, 
                             epoch=ckpt_epoch)
end

function parse_commandline()
    s = ArgParseSettings(autofix_names=true)
    @add_arg_table! s begin
        "--mpi"
            help = "use DistributedUtils/MPI for distributed training"
            action = :store_true
        "--seed", "-s"
            help = "random seed"
            arg_type = Int
            default = rand(1:9999)
        "--train"
            help = "train"
            action = :store_true
        "--train-continue", "-c"
            help = "continue training network"
            action = :store_true
        "--verbose", "-v"
            help = "print extra initialization info"
            action = :store_true
        "--warmup", "-w"
            help = "do train/val warmup"
            action = :store_true
        "--network-config"
            help = "network config file"
            required = false
            default  = "config/cdlnet.yaml"
            arg_type = String
        "--train-config"
            help = "train config file"
            required = false
            default  = "config/train.yaml"
            arg_type = String
        "--closure-config"
            help = "closure config file"
            required = false
            default  = "config/closure.yaml"
            arg_type = String
        "--data-config"
            help = "data config file"
            required = false
            default  = "config/data.yaml"
            arg_type = String
        "--log-config"
            help = "log config file"
            required = false
            default  = "config/log.yaml"
            arg_type = String
        "--config"
            help = "Config file. Overrides all other config options."
            required = false
            default  = missing
            arg_type = Union{String, Missing}
    end
    return parse_args(s; as_symbols=true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline()

    main(; args...)
end
