# The Krylov space: the Arnoldi iteration, as in ReSolver.jl, and the Givens rotations that follow the
# residual of its least-squares problem.
#
# The iteration builds an orthonormal basis Q of the Krylov space of the operator A from b, one vector
# per call to `step!`, and the upper Hessenberg matrix H with A Qₙ = Qₙ₊₁ H.
#
# Interface: the vectors broadcast and support `norm`, `dot` and `similar`; the operator is called
# in place as `A(out, v)`.
#
# The basis is the container of `_basis(x)`, a vector of copies by default. A type of vectors can
# give its own with methods of `_basis`, `_push!`, `_orthogonalise!` and `lincomb!`, and `length`,
# `getindex` and `empty!`, to store them together and orthogonalise a vector against many at once.

_basis(x)            = typeof(x)[]
_push!(Q::Vector, v) = push!(Q, copy(v))

mutable struct ArnoldiIteration{X, B, OP}
    A::OP              # linear operator, called as A(out, v)
    Q::B               # Arnoldi vectors, an empty basis at the start
    q::X               # the last Arnoldi vector
    v::X               # work vector for the next one
    H::Matrix{Float64} # upper Hessenberg matrix, (n + 1) × n after n steps

    # start from b, normalised in place
    function ArnoldiIteration(A::OP, Q::B, b::X) where {X, B, OP}
        b ./= norm(b)
        _push!(Q, b)
        return new{X, B, OP}(A, Q, b, similar(b), zeros(1, 0))
    end
end

# one Arnoldi step: a new basis vector and a new column of H
function step!(arn::ArnoldiIteration)
    Q = arn.Q
    n = length(Q)

    # ---- H grows by one row and one column ----
    H                = zeros(n + 1, n)
    H[1:n, 1:n - 1] .= arn.H
    arn.H            = H

    # ---- new direction, orthogonal to the basis ----
    v = arn.v
    arn.A(v, arn.q)

    _orthogonalise!(v, Q, view(H, 1:n, n))

    # ---- normalisation; a zero vector means breakdown: the Krylov space is invariant ----
    H[n + 1, n] = norm(v)
    H[n + 1, n] > 0 && (v ./= H[n + 1, n])
    _push!(Q, v)

    # ---- the new vector is the next to apply A to, the old one the next work vector ----
    arn.v = arn.q
    arn.q = v

    return Q, H
end

# Orthogonalise v against the basis Q, accumulating the projections in h. Each pass is modified
# Gram–Schmidt, and two passes keep Q orthonormal to rounding when v lies almost in the span of Q
# ("twice is enough", Giraud et al. 2005).
function _orthogonalise!(v::V,
                         Q::Vector{V},
                         h::AbstractVector) where {V}
    for _ in 1:2
        for j in eachindex(h)
            hj    = dot(v, Q[j])
            v   .-= hj .* Q[j]
            h[j] += hj
        end
    end

    return v
end

# the vector Σⱼ yⱼ Qⱼ from its coordinates y on the first length(y) vectors of Q
function lincomb!(out::V,
                    Q::AbstractVector{V},
                    y::AbstractVector) where {V}
    # overwrite with the first term: out may hold uninitialised values, even NaN
    out .= y[1] .* Q[1]
    for j in 2:length(y)
        out .+= y[j] .* Q[j]
    end

    return out
end

# Residual of the least-squares problem min ‖g - H y‖ after a new column h of the upper Hessenberg
# matrix H: the previous Givens rotations are applied to h, a new one zeroes its last entry and
# rotates the right-hand side γ, whose last entry is the residual. O(k) operations per column,
# against O(k³) for solving the problem again.
function _givens!(qr, h::AbstractVector)
    c, s, γ = qr
    k       = length(h) - 1

    # ---- previous rotations ----
    for j in 1:k - 1
        h[j], h[j + 1] = c[j] * h[j] + s[j] * h[j + 1], -s[j] * h[j] + c[j] * h[j + 1]
    end

    # ---- new rotation, and the rotated right-hand side ----
    ρ = hypot(h[k], h[k + 1])
    push!(c, ρ > 0 ? h[k] / ρ : 1.0)
    push!(s, ρ > 0 ? h[k + 1] / ρ : 0.0)
    push!(γ, -s[k] * γ[k])
    γ[k] *= c[k]

    return abs(γ[k + 1])
end
