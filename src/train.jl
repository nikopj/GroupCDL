abstract type AbstractProjection end
struct DefaultProjection <: AbstractProjection
    step::Int
end
project!(net, ps, st, ::DefaultProjection) = project!(net, ps, st)

function select_projection(; proj_type="default", step=1, kws...)
    if ismissing(proj_type) || isnothing(proj_type) || proj_type == "default"
        return DefaultProjection(step)
    else
        throw(ErrorException("proj-type $proj_type not implemented."))
    end
end

function compute_gradnorm(gs)
    gradnorm = 0
    fmap(gs) do ∇
        if !isnothing(∇)
            gradnorm += sum(abs2, ∇)
        end
    end
    sqrt(gradnorm)
end

function clipgrad(gs, clipnorm)
    gradnorm = compute_gradnorm(gs)
    if gradnorm > clipnorm
        gs = fmap(gs) do ∇
            isnothing(∇) ? ∇ : ∇ * clipnorm / gradnorm
        end
    end
    return gs
end

function create_windowsize_sched(; ws_start=1, ws_end=35, epochs_start=1, epochs_end=3000)
    R(x) = floor(Int, x) |> x->iseven(x) ? x + 1 : x
    ws_seq = (ws_start ÷ 2):(ws_end ÷ 2)
    epochs_step = floor(Int, (epochs_end - epochs_start) / length(ws_seq))
    Sequence([R(ws_start), Sequence([R(2w+1) for w=ws_seq], epochs_step*ones(length(ws_seq)))], [epochs_start, epochs_end])
end

function create_sched(; sched_type="cos", lr_start=1e-3, warmup=0, lr_warmup=lr_start, kws...)
    if sched_type == "cos"
        sched = Sequence(
                         [Float32(lr_start),
                          CosAnneal(; λ0=Float32(lr_start),
                                      λ1=Float32(kws[:lr_end]),
                                      period=kws[:epochs_cos],
                                      restart=false),
                          Float32(kws[:lr_end])],
                         [kws[:epochs_start], kws[:epochs_cos], kws[:epochs_end]]
                )
    elseif sched_type == "exp"
        sched = Exp(Float32(lr_start), Float32(kws[:gamma]))
    else
        throw(ErrorException("lr scheduler type=$sched_type not implemented."))
    end
    if warmup > 0
        sched = Sequence(Sin(;
                              λ0 = Float32(lr_warmup),
                              λ1 = Float32(lr_start),
                              period = 2warmup) => warmup,
                         sched => 1_000_000) # arbitrarily long
    end
    return sched
end

function allreduce_gradients!(distr_backend, gs)
    gs = fmap(gs) do ∇
        if !isnothing(∇)
            DistributedUtils.allreduce!(distr_backend, ∇, DistributedUtils.avg)
        end
        ∇
    end
    return gs
end

# Reset the cached adjacency matrix in GroupCDLNet state between epochs/checkpoints.
# For standard CDLNet this is a no-op.
reset_adjacency(tstate::Lux.Training.TrainState) = tstate
function reset_adjacency(tstate::Lux.Training.TrainState{<:GroupCDLNet})
    new_st = Lux.update_state(tstate.states, :Γ, missing)
    return Lux.Training.TrainState(tstate.model, tstate.parameters, new_st, tstate.optimizer_state)
end

function train_epoch!(
        closure::AbstractClosure,
        dl::DataLoader,
        tstate::Lux.Training.TrainState;
        backend::AbstractADType = AutoZygote(),
        desc = "",
        device = identity,
        verbose = true,
        clipnorm = Inf,
        projection::AbstractProjection = DefaultProjection(1),
        maxit_epoch = Inf,
        distr_backend = missing,
        should_log = ()->true,
        is_distributed = ()->false,
        total_workers = 1,
        local_rank = 0,
    )
    tstate = Lux.Training.TrainState(tstate.model,
                                      tstate.parameters,
                                      Lux.trainmode(tstate.states),
                                      tstate.optimizer_state)
    maxit_epoch = min(length(dl), maxit_epoch) |> Int
    lossvec  = Vector{Float32}(undef, maxit_epoch)
    gnormvec = Vector{Float32}(undef, maxit_epoch)
    clipnorm = Float32(clipnorm)

    progmeter = PM.Progress(maxit_epoch;
                            desc=desc,
                            showspeed=true,
                            enabled=(verbose && should_log()))

    for (i, sample) in enumerate(dl)
        i > maxit_epoch && break
        sample = sample |> device

        t = time()
        gs, loss, _, tstate = Lux.Training.compute_gradients(backend, closure, sample, tstate)
        loss_time = time() - t

        # keep losses separate over workers, but reduce to check for NaNs
        loss = if is_distributed()
            MPI.Allreduce(loss, +, MPI.COMM_WORLD) / total_workers
        else
            loss
        end

        if isnan(loss)
            lossvec[i] = NaN
            CUDA.unsafe_free!(sample)
            break
        end

        if is_distributed()
            gs = allreduce_gradients!(distr_backend, gs)
        end
        gs = clipgrad(gs, clipnorm)
        gradnorm = compute_gradnorm(gs)

        if isnan(gradnorm)
            lossvec[i] = NaN
            CUDA.unsafe_free!(sample)
            break
        end

        tstate = Lux.Training.apply_gradients!(tstate, gs)

        if i % projection.step == 0
            project!(tstate.model, tstate.parameters, tstate.states, projection)
        else
            project!(tstate.model, tstate.parameters, tstate.states)
        end

        lossvec[i] = loss
        gnormvec[i] = gradnorm

        CUDA.unsafe_free!(sample)
        GC.safepoint()

        PM.next!(progmeter;
                 showvalues=[(:loss, loss),
                             (:∇norm, gradnorm),
                             (:loss_time, loss_time),
                            ]
                )
    end

    return lossvec, tstate, gnormvec
end

function eval_epoch(
        closure::AbstractClosure,
        dl::DataLoader,
        net,
        ps,
        st::NamedTuple;
        desc = "",
        device = identity,
        logger = missing,
        log_indices = (-1,),
        verbose = true,
        maxit_epoch = Inf,
    )
    st = Lux.testmode(st)
    maxit_epoch = min(length(dl), maxit_epoch) |> Int
    metrics = init_metrics(closure, maxit_epoch)
    progmeter = PM.Progress(maxit_epoch; desc=desc, showspeed=true, enabled=verbose)

    for (i, sample) in enumerate(dl)
        i > maxit_epoch && break
        sample = sample |> device

        sample_metrics, outputs = closure(Val(true), sample, net, ps, st)

        if i in log_indices && !ismissing(logger)
            log_outputs(closure, logger, outputs)
        end

        for k in keys(metrics)
            metrics[k][i] = sample_metrics[k]
        end

        CUDA.unsafe_free!(sample)
        PM.next!(progmeter; showvalues=[(k, metrics[k][i]) for k in keys(metrics)])
    end
    return metrics
end

function mosaic_dictionary(W::AbstractArray)
    mos = cat([mosaicview(W[:,:,c,:];
                          nrow=ceil(Int, sqrt(size(W)[end])),
                          rowmajor=true,
                          npad=1)
               for c=1:size(W,3)]...; dims=3)

    if eltype(mos) <: Complex && size(mos,3) == 1   # Complex 1-channel dict
        mos = cat(real(mos), zeros_like(real(mos)), imag(mos); dims=3)
    end

    if size(mos, 3) > 3
        mos = @view mos[:,:,1:3]
    end

    a, b = extrema(mos)
    return abs.((mos .- a) ./ (b - a))
end
mosaic_dictionary(W::CuArray) = mosaic_dictionary(Array(W))
mosaic_dictionary(net, ps, st) = mosaic_dictionary(conv_dictionary(net, ps, st))

function log_dictionary(logger, net, ps, st)
    mos = mosaic_dictionary(net, ps, st)
    log_image(logger, "dictionary", mos, HWC)
    return mos
end

function train!(
        closure::AbstractClosure,
        dl::NamedTuple,
        net,
        ps,
        st::NamedTuple;
        backend::AbstractADType = AutoZygote(),
        rng = Random.GLOBAL_RNG,
        Δval = 1,
        logger = missing,
        logdir = logger.logdir,
        start = 0,
        end_epoch = 1,
        verbose = true,
        device = identity,
        cpu_dev = cpu_device(),
        backtrack_factor = 100,
        backtrack_count = 0,
        num_log_imgs = 1,
        bestloss = Inf32,
        projection_kws = Dict(:proj_type=>"default"),
        train_epoch_kws = Dict(:maxit_epoch=>Inf, :clipnorm=>Inf32),
        val_epoch_kws = Dict(:maxit_epoch=>Inf),
        test_epoch_kws = Dict(:maxit_epoch=>Inf),
        sched_kws = Dict(:sched_type=>"exp", :lr_start=>1e-3),
        windowsize_sched_kws = Dict(:ws_start=>1, :ws_end=>35),
        distr_backend = missing,
        should_log = ()->true,
        is_distributed = ()->false,
        total_workers = 1,
        local_rank = 0,
        config = missing,
        num_accum_grad = 1,
    )
    verbose = should_log() && verbose
    log_indices = shuffle(rng, 1:min(length(dl.val), val_epoch_kws[:maxit_epoch]))[1:num_log_imgs]
    num_workers = is_distributed() ? total_workers : 1

    best_metrics = Dict(k=>-Inf for k in metric_keys(closure))

    opt = OptimiserChain(AccumGrad(num_accum_grad), Adam())

    if is_distributed()
        @warn "manually averaging gradients. not using distributed optimizer."
    end

    tstate = Lux.Training.TrainState(net, ps, st, opt)

    LR0 = sched_kws[:lr_start]
    sched_kws[:lr_start] = LR0*((0.9)^backtrack_count)
    sched = create_sched(; sched_kws...)
    proj = select_projection(; projection_kws...)

    windowsize_sched = create_windowsize_sched(; windowsize_sched_kws...)
    should_adjust_ws = net isa GroupCDLNet
    @warn "should_adjust_ws=$should_adjust_ws"

    if should_log() && start == 0
        tstate = reset_adjacency(tstate)
        save(joinpath(logdir, "0.bson"),
             :ps => tstate.parameters |> cpu_dev,
             :st => tstate.states |> cpu_dev,
             :lr => sched(1),
             :epoch => 0,
             :backtrack_count => 0,
             :config => config,
             :bestloss => Inf32)
    end

    epoch = start
    log_epoch = start

    if should_log()
        @warn "[$(local_rank+1)/$(total_workers)] backend=$(backend)"
        TensorBoardLogger.set_step!(logger, log_epoch)
        log_dictionary(logger, net, tstate.parameters, tstate.states)
    end

    while epoch < end_epoch
        epoch += 1
        log_epoch += 1
        should_log() && TensorBoardLogger.set_step!(logger, log_epoch)

        Optimisers.adjust!(tstate, sched(epoch))
        if should_adjust_ws
            tstate = Lux.Training.TrainState(tstate.model,
                                              tstate.parameters,
                                              Lux.update_state(tstate.states, :windowsize, windowsize_sched(epoch)),
                                              tstate.optimizer_state)
            should_log() && with_logger(logger) do
                @info "train" windowsize=windowsize_sched(epoch) log_step_increment=0
            end
        end

        should_log() && with_logger(logger) do
            @info "train" backtrack=backtrack_count lr=sched(epoch) log_step_increment=0
        end

        # TRAIN
        lossvec, tstate, gnormvec = train_epoch!(closure, dl.train, tstate;
                                                 backend = backend,
                                                 desc = "TRN-$epoch",
                                                 verbose = verbose,
                                                 device = device,
                                                 projection = proj,
                                                 distr_backend = distr_backend,
                                                 should_log = should_log,
                                                 is_distributed = is_distributed,
                                                 total_workers = total_workers,
                                                 local_rank = local_rank,
                                                 train_epoch_kws...)
        avg_loss = mean(lossvec)
        avg_gnorm = mean(gnormvec)
        if is_distributed()
            avg_loss = MPI.Allreduce(avg_loss, +, MPI.COMM_WORLD) / total_workers
            avg_gnorm = MPI.Allreduce(avg_gnorm, +, MPI.COMM_WORLD) / total_workers
        end
        verbose && println("avg_loss=$(avg_loss), avg_gnorm=$(avg_gnorm)")

        ps, st = tstate.parameters, tstate.states

        # BACKTRACK CHECK
        if avg_loss <= bestloss
            bestloss = avg_loss
            tstate = reset_adjacency(tstate)
            ps, st = tstate.parameters, tstate.states
            should_log() && save(joinpath(logdir, "net_loss.bson"),
                 :ps => ps |> cpu_dev,
                 :st => st |> cpu_dev,
                 :lr => sched(epoch),
                 :epoch => epoch,
                 :backtrack_count => backtrack_count,
                 :config => config,
                 :bestloss => bestloss)

        elseif isnan(avg_loss) || (abs(avg_loss / (1e-16 + bestloss)) > backtrack_factor)
            backtrack_count += 1
            println("loss/best = $(abs(avg_loss))/$(1e-16 + bestloss) = $(abs(avg_loss / (1e-16 + bestloss)))")

            # load checkpoint
            ckpt = load(joinpath(logdir, epoch > 1 ? "net.bson" : "0.bson"))
            ps = ckpt[:ps] |> device
            st = ckpt[:st] |> device
            epoch = ckpt[:epoch]
            bestloss = ckpt[:bestloss]
            if haskey(ckpt[:config], :share)
                share_dict = ckpt[:config][:share]
                ps = GroupCDL.share_parameters(net, ps, share_dict)
            else
                @warn "not resharing parameters as :share key not found!"
            end

            # rebuild TrainState from checkpoint with reduced learning rate
            sched_kws[:lr_start] = LR0*((0.9)^backtrack_count)
            sched = create_sched(; sched_kws...)
            tstate = Lux.Training.TrainState(net, ps, st, opt)
            Optimisers.adjust!(tstate, sched(epoch + 1))

            println("backtracking to epoch $epoch, setting lr to $(sched(epoch+1) * num_workers)")
            continue
        end

        should_log() && with_logger(logger) do
            @info "train" loss=avg_loss grad_norm=avg_gnorm log_step_increment=0
        end

        # VAL
        if epoch % Δval == 0
            metrics = eval_epoch(closure, dl.val, net, ps, st;
                                 desc = "VAL-$epoch",
                                 device = device,
                                 logger = logger,
                                 verbose = verbose,
                                 log_indices = log_indices,
                                 val_epoch_kws...)
            avg_metrics = Dict(k=>mean(metrics[k]) for k in metric_keys(closure))
            if is_distributed()
                avg_metrics = Dict(k=>MPI.Allreduce(avg_metrics[k], +, MPI.COMM_WORLD) / total_workers for k in metric_keys(closure))
            end
            verbose && println("avg_metrics=", avg_metrics)

            should_log() && with_logger(logger) do
                @info "val" metrics=avg_metrics log_step_increment=0
            end

            for k in keys(avg_metrics)
                if avg_metrics[k] >= best_metrics[k]
                    best_metrics[k] = avg_metrics[k]
                    tstate_save = reset_adjacency(tstate)
                    should_log() && save(joinpath(logdir, "net_$k.bson"),
                                         :ps => tstate_save.parameters |> cpu_dev,
                                         :st => tstate_save.states |> cpu_dev,
                                         :lr => sched(epoch),
                                         :epoch => epoch,
                                         :backtrack_count => backtrack_count,
                                         :config => config,
                                         :bestloss => bestloss,
                                         Symbol("best"*string(k)) => best_metrics[k])
                end
            end
            should_log() && log_dictionary(logger, net, ps, st)

        end # end val if

        if is_distributed()
            D = conv_dictionary(net, ps, st)
            meanD = deepcopy(D)
            DistributedUtils.allreduce!(distr_backend, meanD, DistributedUtils.avg)
            @assert D ≈ meanD "Distributed parameters are out of sync!"
        end

        tstate = reset_adjacency(tstate)
        ps, st = tstate.parameters, tstate.states
        should_log() && save(joinpath(logdir, "net.bson"),
                             :ps => ps |> cpu_dev,
                             :st => st |> cpu_dev,
                             :lr => sched(epoch),
                             :epoch => epoch,
                             :backtrack_count => backtrack_count,
                             :config => config,
                             :bestloss => bestloss)
        GC.safepoint()
        GC.gc(true)
        CUDA.reclaim()
    end # end train while

    ps, st = tstate.parameters, tstate.states

    # TEST
    metrics = eval_epoch(closure, dl.test, net, ps, st;
                         desc = "TST-$epoch",
                         verbose = verbose,
                         device = device,
                         test_epoch_kws...)
    avg_metrics = Dict(k=>mean(metrics[k]) for k in metric_keys(closure))
    if is_distributed()
        avg_metrics = Dict(k=>MPI.Allreduce(avg_metrics[k], +, MPI.COMM_WORLD) / total_workers for k in metric_keys(closure))
    end
    verbose && println("avg_metrics=", avg_metrics)

    should_log() && with_logger(logger) do
        @info "test" metrics=avg_metrics log_step_increment=0
    end

    return ps, st
end
