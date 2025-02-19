randnC32(rng::AbstractRNG, size...) = randn(rng, ComplexF32, size...)

function select_network(; network_type="cdlnet", kws...)
    nt = replace(lowercase(network_type), "_"=>"")
    if nt == "cdlnet"
        return CDLNet(; kws...)
    else
        throw(ErrorException("network-type $network_type ($nt) not implemented."))
    end
end

struct DummyNetwork <: LuxCore.AbstractExplicitLayer end
(net::DummyNetwork)(x::AbstractArray, ps, st) = (x, st)
(net::DummyNetwork)((y, H)::Tuple{<:AbstractArray, <:AbstractOperator}, ps, st) = (H'(y), st)

struct CDLNet{L, D, P<:AbstractPreprocess} <: LuxCore.AbstractExplicitContainerLayer{(:lista, :dict)}
    lista::L
    dict::D
    in_chs::Integer
    subbands::Integer
    stride::Integer
    preproc::P
    is_complex::Bool
    function CDLNet(;
            K = 20, 
            M = 32, 
            C = 1, 
            p = 7, 
            s = 1, 
            d = 0, 
            τ0 = 1f-2,
            MoG = 0, 
            use_position = false, # for GaborConv
            is_complex::Bool = false, 
            preproc_type = "image",
            resize_noise = false,
            windowsize = 1,
            kws...
        )

        padl, padr = cld(p-s, 2), fld(p-s, 2)
        pad = (padl, padr, padl, padr)
        init_weight = is_complex ? randnC32 : Lux.randn32

        lista_kws = Dict(:τ0=>τ0, 
                         :degrees=>d, 
                         :pad=>pad, 
                         :stride=>s, 
                         :MoG=>MoG,
                         :use_position=>use_position,
                         :init_weight=>init_weight,
                         :windowsize=>windowsize,)

        lista = LISTA(K, (p, p), C=>M; merge(lista_kws, kws)...)

        if MoG > 0
            dict = GaborConvTranspose((p, p), M=>C; MoG=MoG, is_complex=is_complex, use_bias=false, pad=pad, stride=s, init_weight=init_weight, use_position=use_position)
        else
            dict = Lux.ConvTranspose((p, p), M=>C; use_bias=false, pad=pad, stride=s, init_weight=init_weight)
        end

        preproc = if preproc_type == "kspace"
            KSpacePreprocess(s, resize_noise) 
        elseif preproc_type == "image"
            ImagePreprocess(s, resize_noise)
        elseif preproc_type == "identity"
            IdentityPreprocess()
        else
            throw(ErrorException("preprocessing-type $preproc_type not implemented."))
        end

        return new{typeof(lista), typeof(dict), typeof(preproc)}(lista, dict, C, M, s, preproc, is_complex)
    end
end

const GroupCDLNet = CDLNet{<: GroupLISTA}
iters(net::CDLNet) = iters(net.lista)

function preload(net::CDLNet, ps, pre_net::CDLNet, pre_ps) 
    @reset ps.lista = preload(net.lista, ps.lista, pre_net.lista, pre_ps.lista)
    @reset ps.dict = pre_ps.dict
    return ps
end

function Lux.initialparameters(rng::AbstractRNG, l::CDLNet{L}) where L
    ps_lista = Lux.initialparameters(rng, l.lista)
    ps_dict = Lux.initialparameters(rng, l.dict)
    if l.lista.layer isa LISTALayer
        @reset ps_dict = deepcopy(ps_lista.layer_1.synthesis)
    else
        throw(ErrorException("l.lista.layer=$(l.lista.layer) must be ISTALayer or LISTALayer"))
    end

    return (lista=ps_lista, dict=ps_dict)
end

function project!(net::CDLNet, ps, st::NamedTuple) 
    project!(net.lista, ps.lista, st.lista)
    project!(net.dict, ps.dict, st.dict)
end
conv_dictionary(net::CDLNet, ps, st::NamedTuple) = kernel(net.dict, ps.dict, st.dict)

function (net::CDLNet)((z, (y, H))::Tuple{Tz, Tuple}, ps, st::NamedTuple) where {Tz}
    ỹ, ppt = preprocess(net.preproc, y, H)
    z, st_l = net.lista((z, (ỹ, H)), ps.lista, st.lista)
    x, st_d = net.dict(z, ps.dict, st.dict)
    x = postprocess(net.preproc, x, ppt)
    return x, merge(st, (lista=st_l, dict=st_d, latent=z))
end

function (net::CDLNet)((y, H)::Tuple{Ty, <: AbstractOperator}, ps, st::NamedTuple) where {Ty}
    ỹ, ppt = preprocess(net.preproc, y, H)
    z, st_l = net.lista((ỹ, H), ps.lista, st.lista)
    x, st_d = net.dict(z, ps.dict, st.dict)
    x = postprocess(net.preproc, x, ppt)
    return x, merge(st, (lista=st_l, dict=st_d, latent=z))
end

function share_parameters(network::CDLNet, ps, args...)
    return ps
end

function share_parameters(network::GroupCDLNet, ps, share_dict)
    if share_dict[:αβ]
        @warn "Sharing Wα across layers and Wβ across layers."
        ps = Lux.share_parameters(ps, (["lista.layer_$k.prox.Wα" for k in 1:GroupCDL.iters(network)],
                                       ["lista.layer_$k.prox.Wβ" for k in 1:GroupCDL.iters(network)]))
    end
    if share_dict[:θϕ]
        @warn "Sharing Wθ across layers and Wϕ across layers."
        ps = Lux.share_parameters(ps, (["lista.layer_$k.prox.nlss.Wθ" for k in 1:GroupCDL.iters(network)],
                                       ["lista.layer_$k.prox.nlss.Wϕ" for k in 1:GroupCDL.iters(network)]))
    end
    if share_dict[:γ]
        @warn "Sharing γ across layers."
        ps = Lux.share_parameters(ps, (["lista.layer_$k.prox.γ" for k in 1:GroupCDL.iters(network)],))
    end
    if share_dict[:ρ]
        @warn "Sharing ρ across layers."
        ps = Lux.share_parameters(ps, (["lista.layer_$k.prox.nlss.ρ" for k in 1:GroupCDL.iters(network)],))
    end
    return ps
end

