cisorcos(::Val{false}, x) = cospi(x)
cisorcos(::Val{true}, x)  = cispi(x)

function kernelgrid(ks::Int)
    # input: kernelsize
    # output: (2, ks, ks)
    m, n = ceil(Int, (ks-1)/2), floor(Int, (ks-1)/2)
    u = ones(Float32, ks)
    v = copy(u)
    v .= -m:n
    X = u * v'
    return stack((X, X'), dims=1)
end
CRC.@non_differentiable kernelgrid(::Any...)

function gaborkernel(a::AbstractArray{Ta,6}, f::AbstractArray{Tf,6}, 
        ψ::AbstractArray{Tp,5}, grid::AbstractArray, is_complex::Val=Val(false), pos=zero(Ta)) where {Ta, Tf, Tp}
    """
    a (precision): (2, 1, 1, in_ch, out_ch, batch)
    f (freq):      (2, 1, 1, in_ch, out_ch, batch)
    ψ (phase):        (1, 1, in_ch, out_ch, batch)
    pos (position): (2, 1, 1, in_ch, out_ch, batch)
    output:   (ks, ks, in_ch, out_ch, batch)
    """
    ax = dropdims(sum(abs2, a.*(grid .- pos), dims=1), dims=1)
    fx = dropdims(sum(f.*(grid .- pos), dims=1), dims=1)
    return @. exp(-ax)*cisorcos(is_complex, fx + ψ)
end
function gaborkernel(
        a::AbstractArray{Ta,3}, 
        f::AbstractArray{Tf,3}, 
        ψ::AbstractArray{Tp,1}, 
        grid::AbstractArray, 
        is_complex::Val=Val(false),
        pos = zeros_like(a),
    ) where {Ta, Tf, Tp}

    a = reshape(a, 2, 1, 1, 1, 1, :)
    f = reshape(f, 2, 1, 1, 1, 1, :)
    ψ = reshape(ψ, 1, 1, 1, 1, :)
    pos = reshape(pos, size(a))

    return gaborkernel(a, f, ψ, grid, is_complex, pos)
end
gaborkernel(a, f, ψ, ks::Int, is_complex::Val=Val(false), pos=0f0) = gaborkernel(a, f, ψ, kernelgrid(ks), is_complex)

const ConvOrConvT = Union{Lux.Conv, Lux.ConvTranspose}

struct GaborConvLayer{L <: ConvOrConvT, is_complex, use_position} <: Lux.AbstractExplicitLayer
    layer::L
    MoG::Int
end

function Base.show(io::IO, l::GaborConvLayer{L}) where L
    name = L <: Lux.Conv ? "GaborConv" : "GaborConvTranspose"
    print(io, "$(name)($(l.kernel_size), $(l.in_chs)=>$(l.out_chs), stride=$(l.stride), MoG=$(l.MoG)", 
          l.is_complex ? ", is_complex=true" : "",
          l.use_position ? ", use_position=true" : "",
          ")",
         )
end

function Base.getproperty(l::GaborConvLayer{L, is_complex, use_position}, s::Symbol) where {L, is_complex, use_position}
    if s == :MoG || s == :layer 
        return Base.getfield(l, s)
    elseif s == :is_complex
        return is_complex == Val(true)
    elseif s == :use_position
        return use_position 
    end
    return Base.getfield(l.layer, s)
end

const GaborConv = GaborConvLayer{<: Lux.Conv}
const GaborConvTranspose = GaborConvLayer{<: Lux.ConvTranspose}

function GaborConv(args...; is_complex=false, MoG=1, use_position=false, kws...)
    C = Lux.Conv(args...; kws...)
    return GaborConvLayer{typeof(C), Val(is_complex), use_position}(C, MoG)
end

function GaborConvTranspose(args...; is_complex=false, MoG=1, use_position=false, kws...)
    C = Lux.ConvTranspose(args...; kws...)
    return GaborConvLayer{typeof(C), Val(is_complex), use_position}(C, MoG)
end

function Lux.initialparameters(rng::AbstractRNG, g::GaborConvLayer)
    ps = (scale     = randn(rng, Float32, 1, 1, g.in_chs, g.out_chs, g.MoG),
          precision = randn(rng, Float32, 2, 1, 1, g.in_chs, g.out_chs, g.MoG),
          frequency = randn(rng, Float32, 2, 1, 1, g.in_chs, g.out_chs, g.MoG),
          phase     = randn(rng, Float32, 1, 1, g.in_chs, g.out_chs, g.MoG),
    )
    if g.use_position
        ps = merge(ps, (position=zeros(Float32, 2, 1, 1, g.in_chs, g.out_chs, g.MoG),))
    end
    return ps
end
Lux.initialstates(::AbstractRNG, g::GaborConvLayer) = (kernelgrid=kernelgrid(g.kernel_size[1]),)

function Lux.parameterlength(g::GaborConvLayer)
    if g.use_position 
        return 8*g.MoG*g.in_chs*g.out_chs
    end
    return 6*g.MoG*g.in_chs*g.out_chs
end

@inline function kernel(g::GaborConvLayer{L, is_complex, true} , ps, st) where {L, is_complex}
    W = ps.scale .* gaborkernel(ps.precision, ps.frequency, ps.phase, st.kernelgrid, is_complex, ps.position)
    return dropdims(sum(W, dims=5), dims=5)
end

@inline function kernel(g::GaborConvLayer{L, is_complex, false}, ps, st) where {L, is_complex} 
    W = ps.scale .* gaborkernel(ps.precision, ps.frequency, ps.phase, st.kernelgrid, is_complex)
    return dropdims(sum(W, dims=5), dims=5)
end

@inline function (g::GaborConvLayer)(x, ps, st) 
   return g.layer(x, (weight=kernel(g, ps, st),), st)
end

@inline function kernel(l::Union{Lux.Conv, Lux.ConvTranspose}, ps, st)
    return ps.weight
end

function project!(l::GaborConvLayer, ps, st::NamedTuple)
    W = kernel(l, ps, st)
    Wnorm = sqrt.(sum(abs2, W; dims=1:ndims(W)-2))
    @. ps.scale /= max(1f0, Wnorm)
    if l.use_position
        K = minimum(l.kernel_size) / 4
        @. ps.position = clamp(ps.position, -K, K)
    end
    return ps
end
