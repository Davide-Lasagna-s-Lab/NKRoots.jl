# The right preconditioner: its linearisation, the metric of the step it gives, and the operator of
# flexible GMRES.

export update!,
       gram

"""
    update!(J, x, r)
    update!(M, x)

Linearise the Jacobian `J` at `x`, with `r = F(x)`, and the preconditioner `M` at `x`. The
defaults do nothing, for operators that do not depend on `x`.
"""
update!(J, x, r) = J
update!(M, x)    = M

"""
    gram(M, Q) -> Matrix or UpperTriangular

The Gram matrix `G[i, j] = dot(M⁻¹ Q[i], M⁻¹ Q[j])` of the orthonormal Krylov vectors `Q`, which
gives the length of a step of the preconditioned hookstep, or an `UpperTriangular` `R` with
`Rᵀ R = G`. This default applies `M⁻¹` to each vector and returns `G`; a preconditioner can define
a cheaper method. A Gram matrix loses the short vectors when `M⁻¹` stretches a few directions far.
`R` keeps them if it is computed without forming `G`, and so does `flexible=true`, which does not
call `gram`.
"""
function gram(M, Q)
    Z = [ldiv!(similar(q), M, q) for q in Q]
    return [dot(Z[i], Z[j]) for i in eachindex(Z), j in eachindex(Z)]
end

# what gram gives, as _root takes it
_metric(G::UpperTriangular) = G
_metric(G::AbstractMatrix)  = Symmetric(G)

# the applications of M⁻¹ made by gram: m for the default, none for a preconditioner with its own
_gram_applications(M, m) = which(gram, Tuple{typeof(M), Any}).module === NKRoots ? m : 0

# The Arnoldi operator of flexible GMRES: J z into out, with z = M⁻¹ v. Then z, orthonormalised
# against the vectors before it, goes into W, and its coordinates on them, a column of R, into Rc,
# so that Zₙ = Wₙ R. Gram-Schmidt is applied twice, so that R keeps the precision of the vectors
# when their lengths spread over orders of magnitude, as they do when M⁻¹ solves a few slow
# directions exactly: their inner products would square that spread, and the short vectors would
# fall below rounding. W replaces the vectors zⱼ, with no extra storage, and z is a work vector
function _flexible!(out, product!, precondition!, v, W, Rc, z=similar(v))
    # ---- z = M⁻¹ v, and J z ----
    precondition!(z, v)
    product!(out, z)

    # ---- z orthogonalised against W in place, twice, and normalised ----
    c = zeros(length(W) + 1)
    _orthogonalise!(z, W, view(c, 1:length(W)))
    c[end] = norm(z)
    c[end] > 0 && (z ./= c[end])

    _push!(W, z)
    push!(Rc, c)

    return out
end
