struct NonLocalSimilarity{ρinv, S, W, P} <: Lux.AbstractExplicitContainerLayer{(:Wθ, :Wϕ, :ρ)}
    simfun::S
    nheads::Int
    Wθ::W
    Wϕ::W
    ρ::P
    init_windowsize::Int

    function NonLocalSimilarity(ks::NTuple{N, Integer}, ch::Pair; 
            similarity = "distance",
            learned = false, 
            windowsize = 15,
            nheads = 1,
            degrees = 0,
            ρ0 = 1f0,
            ρinv = true,
            kws...
        ) where N

        if learned
            Wθ = Lux.Conv(ks, ch; use_bias=false, groups=nheads, kws...)
            Wϕ = Lux.Conv(ks, ch; use_bias=false, groups=nheads, kws...)
        else
            Wθ = Lux.NoOpLayer()
            Wϕ = Lux.NoOpLayer()
        end

        simfun = if similarity == "dot"
            DotSimilarity()
        elseif similarity == "distance"
            DistanceSimilarity()
        else
            throw(ErrorException("similarity-type $similarity not implemented."))
        end

        M = learned ? last(ch) : first(ch) 
        ρ = Polynomial(2, M; τ0=Float32(ρ0), degrees=degrees)

        new{ρinv, typeof(simfun), typeof(Wθ), typeof(ρ)}(simfun, nheads, Wθ, Wϕ, ρ, windowsize)
    end
end
NonLocalSimilarity(; ks=(1,1), ch=64=>32, kws...) = NonLocalSimilarity(ks, ch; kws...)

function Lux.initialstates(rng::AbstractRNG, l::NonLocalSimilarity)
    st_t = Lux.initialstates(rng, l.Wθ)
    st_p = Lux.initialstates(rng, l.Wϕ)
    st_r = Lux.initialstates(rng, l.ρ)
    (Wθ=st_t, Wϕ=st_p, ρ=st_r, windowsize=l.init_windowsize)
end

function project!(nls::NonLocalSimilarity, ps, st::NamedTuple) 
    clamp!(ps.ρ.weight, 0.1f0, Inf32)
end

function (l::NonLocalSimilarity{true})((x, y, σ)::Tuple{T,T,Ts}, ps, st::NamedTuple) where {T, Ts}
    xθ, st_θ = l.Wθ(x, ps.Wθ, st.Wθ)
    yϕ, st_ϕ = l.Wϕ(y, ps.Wϕ, st.Wϕ)

    ρ, st_ρ  = l.ρ(σ, ps.ρ, st.ρ)
    sqρ = @. sqrt(ρ + 1f-8) 

    Γ = circulant_mh_adjacency(l.simfun, 
                               xθ ./ (sqρ .+ 1f-8), 
                               yϕ ./ (sqρ .+ 1f-8), 
                               st.windowsize, 
                               l.nheads)
    return Γ, merge(st, (Wθ=st_θ, Wϕ=st_ϕ, ρ=st_ρ))
end
function (l::NonLocalSimilarity{false})((x, y, σ)::Tuple{T,T,Ts}, ps, st::NamedTuple) where {T, Ts}
    xθ, st_θ = l.Wθ(x, ps.Wθ, st.Wθ)
    yϕ, st_ϕ = l.Wϕ(y, ps.Wϕ, st.Wϕ)

    ρ, st_ρ  = l.ρ(σ, ps.ρ, st.ρ)
    sqρ = @. sqrt(ρ + 1f-8) 

    Γ = circulant_mh_adjacency(l.simfun, 
                               xθ .* sqρ, 
                               yϕ .* sqρ, 
                               st.windowsize, 
                               l.nheads)
    return Γ, merge(st, (Wθ=st_θ, Wϕ=st_ϕ, ρ=st_ρ))
end
(l::NonLocalSimilarity)((x, σ)::Tuple{Tx,Ts}, ps, st::NamedTuple) where {Tx, Ts} = l((x,x,σ), ps, st)

struct GroupThreshold{P1, P2, Wa, Wb, S} <: Lux.AbstractExplicitContainerLayer{(:τ, :γ, :Wα, :Wβ, :nlss)}
    τ::P1
    γ::P2
    Wα::Wa
    Wβ::Wb
    nlss::S
    ΔK::Int

    function GroupThreshold(M::Int; 
        Mh=nothing, 
        nheads=1, 
        degrees=0, 
        ΔK=1, 
        τ0=1f-3,
        γ0=0.8f0,
        ρ0=1f0,
        γ_degrees=0,
        ρ_degrees=0,
        similarity="distance",
        windowsize=15,
        ρinv = true,
    )
        @warn "not using nheads in GT αβ"
        if !isnothing(Mh)
            Wα = Lux.Conv((1,1), M=>Mh; use_bias=false)
            Wβ = Lux.ConvTranspose((1,1), Mh=>M; use_bias=false)
        else
            Wα = Lux.NoOpLayer()
            Wβ = Lux.NoOpLayer()
        end

        τ = Polynomial(2, M; τ0=Float32(τ0), degrees=degrees)
        γ = Polynomial(2, nheads; τ0=Float32(γ0), degrees=γ_degrees)
        nlss = NonLocalSimilarity((1,1), M=>Mh; similarity=similarity, nheads=nheads, windowsize=windowsize, learned=!isnothing(Mh), degrees=ρ_degrees, ρ0=ρ0, ρinv=ρinv)
        new{typeof(τ), typeof(γ), typeof(Wα), typeof(Wβ), typeof(nlss)}(τ, γ, Wα, Wβ, nlss, ΔK)
    end
end

function Lux.initialparameters(rng::AbstractRNG, l::GroupThreshold)
    psα = Lux.initialparameters(rng, l.Wα)
    psγ = Lux.initialparameters(rng, l.γ)
    psτ = Lux.initialparameters(rng, l.τ)
    ps_nlss = Lux.initialparameters(rng, l.nlss)

    if !(l.Wα isa Lux.NoOpLayer)
        W = rand_like(psα.weight)
        psα.weight .= W ./ (opnorm(W[1,1,:,:]) .+ 1f-8)
        psβ = deepcopy(psα)
    else
        psβ = deepcopy(psα)
    end

    # @reset ps_nlss.Wθ = deepcopy(psα)
    @reset ps_nlss.Wϕ = deepcopy(ps_nlss.Wθ)

    (Wα=psα, Wβ=psβ, γ=psγ, τ=psτ, nlss=ps_nlss)
end

function Lux.initialstates(rng::AbstractRNG, l::GroupThreshold)
    st_α = Lux.initialstates(rng, l.Wα)
    st_β = Lux.initialstates(rng, l.Wβ)
    st_τ = Lux.initialstates(rng, l.τ)
    st_γ = Lux.initialstates(rng, l.γ)
    st_nlss = Lux.initialstates(rng, l.nlss)
    (Wα=st_α, Wβ=st_β, τ=st_τ, γ=st_γ, nlss=st_nlss, Γ=missing, Δupdate=1)
end

function project!(gt::GroupThreshold, ps, st::NamedTuple)
    clamp!(ps.τ.weight, 0f0, Inf32)
    clamp!(ps.γ.weight, 0.05f0, 0.95f0)

    if !(gt.Wα isa Lux.NoOpLayer)
        clamp!(ps.Wβ.weight, 0f0, Inf32)
    end

    project!(gt.nlss, ps.nlss, st.nlss)
end

function (gt::GroupThreshold)((z, σ)::Tuple{Tz, Ts}, ps, st::NamedTuple) where {Tz, Ts}
    if ismissing(st.Γ)
        Γ, st_nlss = gt.nlss((z, σ), ps.nlss, st.nlss)
        st = merge(st, (nlss=st_nlss, Γ=Γ, Δupdate=1))

    elseif st.Δupdate % gt.ΔK == 0
        Γnew, st_nlss = gt.nlss((z, σ), ps.nlss, st.nlss)
        γ, st_γ  = gt.γ(σ, ps.γ, st.γ)
        Γ  = st.Γ + γ * (Γnew - st.Γ)
        st = merge(st, (nlss=st_nlss, Γ=Γ, γ=st_γ))
    end

    st = merge(st, (Δupdate=(st.Δupdate + 1) % gt.ΔK, ))
    return gt((z, σ, st.Γ), ps, st)
end

function (gt::GroupThreshold)((z, σ, Γ)::Tuple{Tz, Ts, <:Circulant}, ps, st::NamedTuple) where {Tz, Ts}
    zα, st_α = gt.Wα(z, ps.Wα, st.Wα)
    ξα2      = Γ ⨷ abs2.(zα)
    ξ2       = @. sqrt(ξα2 + 1f-8)
    ξ, st_β  = gt.Wβ(ξ2, ps.Wβ, st.Wβ)

    τ, st_p = gt.τ(σ, ps.τ, st.τ)
    znew    = @. z * relu(1f0 - (τ / (ξ + 1f-8)))
    return znew, merge(st, (Wα=st_α, Wβ=st_β, τ=st_p))
end

const GroupLISTALayer{C, Ct} = LISTALayer{C, Ct, <:GroupThreshold}
const GroupLISTA{K} = LISTA{K, <:GroupLISTALayer}

function (l::GroupLISTA{K})(x::Tuple{T, <: AbstractOperator}, ps, st::NamedTuple) where {K, T}
    st = Zygote.@ignore Lux.update_state(st, :Γ, missing)
    z, st = l.layer(x, ps[1], st)
    for k in 2:K
        z, st = l.layer((z, x), ps[k], st)
    end
    return z, st
end
