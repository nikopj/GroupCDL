struct LISTALayer{C, Ct, P} <: Lux.AbstractExplicitContainerLayer{(:analysis, :synthesis, :prox)}
    analysis::C
    synthesis::Ct
    prox::P
end

function LISTALayer(
        ks::NTuple{N, Integer}, 
        ch::Pair{<:Integer, <:Integer}; 
        τ0=1e-2, 
        degrees=0, 
        stride=1,
        pad=1,
        init_weight=missing,
        MoG=0,
        use_position=false, # for GaborConv
        windowsize=1,
        kws...) where N

    (A, B) = if MoG > 0
        is_complex = eltype(init_weight(Random.default_rng(), 1)) <: Complex ? true : false
        (GaborConv(ks, ch; is_complex=is_complex, MoG=MoG, pad=pad, stride=stride, use_bias=false, use_position=use_position),
         GaborConvTranspose(ks, reverse(ch); is_complex=is_complex, MoG=MoG, pad=pad, stride=stride, use_bias=false, use_position=use_position))
    else
        (Lux.Conv(ks, ch; use_bias=false, stride=stride, pad=pad, init_weight=init_weight),
         Lux.ConvTranspose(ks, reverse(ch); use_bias=false, stride=stride, pad=pad, init_weight=init_weight))
    end

    prox = if windowsize > 1
        GroupThreshold(last(ch); τ0=τ0, degrees=degrees, windowsize=windowsize, kws...)
    else
        SoftThreshold(N, last(ch); τ0=τ0, degrees=degrees)
    end

    LISTALayer{typeof(A), typeof(B), typeof(prox)}(A, B, prox)
end

function Lux.initialparameters(rng::AbstractRNG, l::LISTALayer) 
    ps_a = Lux.initialparameters(rng, l.analysis)
    ps_p = Lux.initialparameters(rng, l.prox)
    st = Lux.initialstates(rng, l.analysis)

    T = if l.analysis isa GaborConv
        l.analysis.is_complex ? ComplexF32 : Float32
    else
        eltype(ps_a.weight)
    end

    z0 = randn(T, ntuple(_->128, length(l.synthesis.kernel_size))..., l.synthesis.in_chs, 1)
    λ, _, flag = powermethod(z0; verbose=false, maxit=500) do z
        l.analysis(l.synthesis(z, ps_a, st)[1], ps_a, st)[1]
    end

    (real(λ) <= 0) && @warn "powermethod: λ=$λ should be positive."
    (imag(λ) > 1e-3) && @warn "powermethod: λ=$λ should real."
    flag && @warn "powermethod: λ=$λ did not converge."
    L = sqrt(Float32(real(λ)))

    if l.analysis isa GaborConv
        ps_a.scale ./= L
    else
        ps_a.weight ./= L
    end

    return (analysis=ps_a, synthesis=deepcopy(ps_a), prox=ps_p)
end

function project!(l::ConvOrConvT, ps, st::NamedTuple)
    W = ps.weight
    Wnorm = sqrt.(sum(abs2, W; dims=1:ndims(W)-2))
    @. ps.weight /= max(1f0, Wnorm)
    return ps
end

function project!(l::LISTALayer, ps, st::NamedTuple)
    project!(l.analysis, ps.analysis, st.analysis)
    project!(l.synthesis, ps.synthesis, st.synthesis)
    project!(l.prox, ps.prox, st.prox)
end

function (l::LISTALayer)((z, π0, (ỹ, H))::Tuple{Tz, Tz, Tuple}, ps, st::NamedTuple) where {Tz}
    σ = Zygote.@ignore H.noise_level
    Bz, st_s = l.synthesis(z, ps.synthesis, st.synthesis)
    Ar, st_a = l.analysis(gramian(H)(Bz) - ỹ, ps.analysis, st.analysis)
    znew, st_p = l.prox((z - Ar, σ), ps.prox, st.prox)
    return znew, merge(st, (analysis=st_a, synthesis=st_s, prox=st_p))
end

function (l::LISTALayer)((z, (ỹ, H))::Tuple{Tz, Tuple}, ps, st::NamedTuple) where {Tz}
    σ = Zygote.@ignore H.noise_level
    Bz, st_s = l.synthesis(z, ps.synthesis, st.synthesis)
    Ar, st_a = l.analysis(gramian(H)(Bz) - ỹ, ps.analysis, st.analysis)
    znew, st_p = l.prox((z - Ar, σ), ps.prox, st.prox)
    return znew, merge(st, (analysis=st_a, synthesis=st_s, prox=st_p))
end

function (l::LISTALayer)((ỹ, H)::Tuple{Ty, <:AbstractOperator}, ps, st::NamedTuple) where {Ty}
    σ = Zygote.@ignore H.noise_level
    Ar, st_a = l.analysis(ỹ, ps.analysis, st.analysis)
    z, st_p = l.prox((Ar, σ), ps.prox, st.prox)
    return z, merge(st, (analysis=st_a, prox=st_p))
end

struct LISTA{K, T} <: Lux.AbstractExplicitContainerLayer{(:layer,)}
    layer::T
    function LISTA(iters::Integer, args...; tol=1e-3, maxit=100, jacobian_free=false, kws...) 
        layer = LISTALayer(args...; kws...)
        new{iters, typeof(layer)}(layer)
    end
end

function Lux.initialparameters(rng::AbstractRNG, l::LISTA{K}) where {K}
    names = ntuple(k->Symbol("layer_$k"), Val(K))
    ps = Lux.initialparameters(rng, l.layer)
    params = ntuple(k->deepcopy(ps), Val(K))
    return NamedTuple{names}(params)
end
Lux.initialstates(rng::AbstractRNG, l::LISTA) = Lux.initialstates(rng, l.layer)
Lux.parameterlength(l::LISTA{K}) where K = K * Lux.parameterlength(l.layer)
iters(::LISTA{K}) where K = K

function project!(l::LISTA{K}, ps, st::NamedTuple) where K
    for k=1:K
        project!(l.layer, ps[k], st)
    end
end

function (l::LISTA{K})(x::Tuple{T, <: AbstractOperator}, ps, st::NamedTuple) where {K, T}
    z, st = l.layer(x, ps[1], st)
    for k in 2:K
        z, st = l.layer((z, x), ps[k], st)
    end
    return z, st
end

function (l::LISTA{K})((z, x)::Tuple{<:AbstractArray, <:Tuple}, ps, st::NamedTuple) where K
    for k in 1:K
        z, st = l.layer((z, x), ps[k], st)
    end
    return z, st
end
(l::LISTA{K})((z, x)::Tuple{<:Missing, Tx}, ps, st::NamedTuple) where {K, Tx} = l(x, ps, st)


function (l::LISTA{K})((z, π0, x)::Tuple{Tz, Tz, Tuple}, ps, st::NamedTuple) where {K, Tz <: AbstractArray}
    for k in 1:K
        z, st = l.layer((z, π0, x), ps[k], st)
    end
    return z, st
end
(l::LISTA{K})((z, π0, x)::Tuple{<:Missing, <:Missing, Tx}, ps, st::NamedTuple) where {K, Tx} = l(x, ps, st)
(l::LISTA{K})((z, π0, x)::Tuple{<:AbstractArray, <:Missing, Tx}, ps, st::NamedTuple) where {K, Tx} = l((z,x), ps, st)

