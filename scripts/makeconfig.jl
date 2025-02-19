using YAML
using Base.Iterators: product

loadyaml(fn) = YAML.load_file(fn; dicttype=Dict{Symbol,Any})
saveyaml(fn::String, d::Dict{Symbol,Any}) = YAML.write_file(fn, d)

function loadconfig(fn::Union{String, Missing}; 
        network = "config/cdlnet.yaml", 
        train = "config/train.yaml", 
        closure = "config/synthawgn_closure.yaml",
        log = "config/log.yaml",
        data = "config/image_data.yaml")
    if ismissing(fn)
        network_d = loadyaml(network)
        train_d = loadyaml(train)
        close_d = loadyaml(closure)
        data_d = loadyaml(data)
        log_d = loadyaml(log)
        return merge(network_d, train_d, close_d, data_d, log_d)
    end
    return loadyaml(fn)
end
loadconfig() = loadconfig(missing)

function makeconfigs(loopd::Dict, fn=missing; name=missing, startnum=1, savedir="hpc/config.d", joint=false, loadkws...)
    based = loadconfig(fn; loadkws...)

    if ismissing(name)
        name = join(keys(loopd), "_")
    end

    summary_fn = joinpath(savedir, "summary_"*name*".txt")
    io = open(summary_fn, "a")

    newds = factory(based, loopd; io=io, joint=joint)

    for nd in newds
        if (nd[:train][:end_epoch] < 100) 
            @warn "end_epoch=$(nd[:train][:end_epoch]), are you sure you know what you're doing?"
        end
        if (nd[:train][:Δval] < 10)
            @warn "Δval=$(nd[:train][:Δval]), are you sure you know what you're doing?"
        end
    end

    for i = 1:length(newds)
        ver = string(startnum + i - 1)
        namei = name * "-" * ver
        newds[i][:log][:logdir] = joinpath("trained_nets", namei)
        println(io, namei)
        println(namei)
        !ismissing(savedir) && saveyaml(joinpath(savedir, namei*".yaml"), newds[i])
    end

    close(io)
    return newds
end

function factory(based::Dict, loopd::Dict; io=stdout, joint=false)
    if joint
        l = length.(values(loopd))
        @assert all(l .== first(l)) "length of all loop vectors must match, got l=$l"
        iter = zip([loopd[k] for k in keys(loopd)]...)
    else
        iter = product([loopd[k] for k in keys(loopd)]...)
    end
    newdvec = []
    println(io, keys(loopd))
    io != stdout && println(keys(loopd))
    for t in iter
        println(io, t)
        io != stdout && println(t)
        newd = deepcopy(based)
        for (k, v) in zip(keys(loopd), t)
            setrecursive!(newd, k, v) && begin @warn "makeconfigs: did not find key $k"; return end
        end
        push!(newdvec, newd)
    end
    return newdvec
end

function setrecursive!(d::Dict, key, value)
    if key in keys(d)
        d[key] = value
        return false
    end
    flag = true
    for k in keys(d)
        if d[k] isa Dict
            flag = setrecursive!(d[k], key, value)
            !flag && break # end greedy search
        end
    end
    return flag
end

function setrecursive!(d::Dict, keyset::Tuple, value)
    key, remaining_keys = first(keyset), Base.tail(keyset)

    if length(remaining_keys) == 0
        return setrecursive!(d, key, value)
    end
    
    # Check if the current key leads to a nested dictionary
    if key in keys(d) && d[key] isa Dict
        return setrecursive!(d[key], remaining_keys, value)
    end

    # If the current key is not a dictionary, return true (not found)
    return true
end

function setrecursive(d, k, v) 
    newd = copy(d)
    setrecursive!(newd, k, v)
    return newd
end

