magnitude_ssim_loss_fast(x, y; kws...) = SSIMLoss.ssim_loss_fast(abs.(1f-7 .+ x), abs.(1f-7 .+ y); kws...)
function SSIMLoss.ssim_loss_fast(x::AbstractArray{T}, y::AbstractArray{T}; kws...) where T<:Complex 
    SSIMLoss.ssim_loss_fast(cat(reim(x)...; dims=3), cat(reim(y)...; dims=3); kws...)
end
magnitude_mse_loss(x, y) = mean(abs2, @. abs(x) - abs(y))
magnitude_l1_loss(x, y) = mean(abs, @. abs(x) - abs(y))

mse_loss(x, y) = mean(abs2, x - y)
nmse_loss(x, y) = mean(mean(abs2, x - y; dims=1:3) ./ mean(abs2, x; dims=1:3))

l1_loss(x, y) = mean(@. abs(x - y + 1f-8))
nl1_loss(x, y) = mean(mean(abs.(x .- y .+ 1f-8); dims=1:3) ./ mean(abs.(x .+ 1f-8); dims=1:3))

l1_ssim_loss(x, y; kws...) = l1_loss(x, y) + SSIMLoss.ssim_loss_fast(x, y; kws...)
magnitude_l1_ssim_loss(x, y; kws...) = begin
    mx, my = abs.(x .+ 1f-8), abs.(y .+ 1f-8)
    l1_loss(mx, my) + SSIMLoss.ssim_loss_fast(mx, my; kws...)
end

l2_loss(x, y) = mean(sqrt.(mean(abs2, x - y; dims=1:3)))
nl2_loss(x, y) = mean(sqrt.(mean(abs2, x - y; dims=1:3) ./ mean(abs2, x; dims=1:3)))
nl1_nl2_loss(x, y) = nl1_loss(x, y) + nl2_loss(x, y)

magnitude_nl1_nl2_loss(x, y) = begin
    mx, my = abs.(x .+ 1f-8), abs.(y .+ 1f-8)
    nl1_nl2_loss(mx, my)
end

function select_loss(loss_type::String)
    t = lowercase(loss_type)
    loss = if t == "mse"
        mse_loss
    elseif t == "l1"
        l1_loss
    elseif t == "mag_ssim"
        magnitude_ssim_loss_fast
    elseif t == "mag_mse"
        magnitude_mse_loss
    elseif t == "mag_l1"
        magnitude_l1_loss
    elseif t == "ssim"
        ssim_loss_fast
    elseif t == "l1_ssim"
        l1_ssim_loss
    elseif t == "mag_l1_ssim"
        magnitude_l1_ssim_loss
    elseif t == "l2"
        l2_loss
    elseif t == "nl2"
        nl2_loss
    elseif t == "nl1"
        nl1_loss
    elseif (t == "nl1_nl2") || (t == "nl2_nl1")
        nl1_nl2_loss
    elseif (t == "mag_nl1_nl2") || (t == "mag_nl2_nl1")
        magnitude_nl1_nl2_loss
    elseif t == "nmse"
        nmse_loss
    else
        throw(ErrorException("loss-type $loss_type not implemented."))
    end
    return loss
end

