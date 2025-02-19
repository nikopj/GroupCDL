function tensor_evalpoly(x::AbstractArray{Tx,N}, c::AbstractArray{Tc, N}) where{Tx, Tc, N}
    y = selectdim(c, N, 1) .* ones_like(x)
    for d in 2:size(c, N)
        y = y + selectdim(c, N, d) .* x.^(d-1)
    end
    return y
end

# soft thresholding
function ST(x::Tx, t::Tt)::Tx where {Tx, Tt}
    @. sign(x)*relu(abs(x + 1f-7) - t)
end

struct SoftThreshold{N, C, d} <: Lux.AbstractExplicitLayer
    init_weight
    function SoftThreshold(spatdims::Integer, chans::Integer; τ0=1e-2, degrees=0, T=Float32) 
        init_weight = () -> begin
            w = zeros(T, (ones(Int, spatdims)..., chans, degrees+1))
            selectdim(w, ndims(w), 1) .= T(τ0)
            return w
        end
        new{spatdims, chans, degrees}(init_weight)
    end
end
(l::SoftThreshold)(x::AbstractArray, ps, st::NamedTuple) = (ST(x, ps.weight), st)
(l::SoftThreshold)((x, σ)::Tuple, ps, st::NamedTuple) = (ST(x, tensor_evalpoly(σ, ps.weight)), st)
(l::SoftThreshold{N,C,0})((x, σ)::Tuple, ps, st::NamedTuple) where {N,C} = (ST(x, ps.weight), st)

LuxCore.initialparameters(rng::AbstractRNG, l::SoftThreshold) = (weight=l.init_weight(),)
LuxCore.parameterlength(l::SoftThreshold{N,C,d}) where {N,C,d} = C*(d+1)

function project!(l::SoftThreshold, ps, st::NamedTuple)
    clamp!(ps.weight, 0f0, Inf32)
end

function Base.show(io::IO, l::SoftThreshold{N,C,d}) where {N,C,d}
    print(io, "SoftThreshold(spatial_dims=$N, channels=$C, degrees=$d)")
end

struct Polynomial{N, C, d} <: LuxCore.AbstractExplicitLayer
    init_weight
    function Polynomial(spatdims::Integer, chans::Integer; τ0=1e-2, degrees=0, T=Float32) 
        init_weight = () -> begin
            w = zeros(T, (ones(Int, spatdims)..., chans, degrees+1))
            selectdim(w, ndims(w), 1) .= T(τ0)
            return w
        end
        new{spatdims, chans, degrees}(init_weight)
    end
end
(l::Polynomial)(x, ps, st::NamedTuple) = (tensor_evalpoly(x, ps.weight), st)
(l::Polynomial{N, C, 0})(x, ps, st::NamedTuple) where {N,C} = (ps.weight, st)

LuxCore.initialparameters(rng::AbstractRNG, l::Polynomial) = (weight=l.init_weight(),)
LuxCore.parameterlength(l::Polynomial{N,C,d}) where {N,C,d} = C*(d+1)

function Base.show(io::IO, l::Polynomial{N,C,d}) where {N,C,d}
    print(io, "Polynomial(spatial_dims=$N, channels=$C, degrees=$d)")
end

