function powermethod(A::Function, b::AbstractArray; maxit=100, tol=1e-3, verbose=true)
    r = zeros(maxit)
    λ, λᵏ= 0, 0
    flag = true         # error flag: tolerance not reached
    for k in 1:maxit
        Ab = A(b)
        λ = sum(conj(b).*Ab) / sum(abs2, b)
        b = Ab ./ norm(b)
        r[k] = abs(λ-λᵏ) / abs(λ + 1e-16)
        λᵏ = λ
        if verbose
            @printf "k: %3d, |λ-λᵏ|= %.3e, Re(λ)=%.3e, Im(λ)=%.3e \n" k r[k] real(λ) imag(λ)
        end
        if r[k] <= tol
            flag = false
            break
        end
    end
    return λ, b, flag
end

