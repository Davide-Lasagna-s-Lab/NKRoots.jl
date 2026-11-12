# Any type that broadcasts. The Bratu problem on the unit square, Δu + exp(u) = 0 with u = 0 on the
# boundary, by second differences on 20 × 20 points, with the unknown kept as a matrix. The solver
# needs broadcasting, similar, copy, dot and norm of the unknown, and nothing else.
#
# Settings: 20 × 20 points, guess zero, finite-difference Jacobian, no preconditioner, Δ = 1,
# krylov_dim = 120, krylov_tol = 1e-6, tol = 1e-6.
#
#     julia --project=examples examples/vectors/vectors.jl
using LinearAlgebra
using NKRoots


# ---- the residual, on a matrix of unknowns ----
const m = 20
const k = 1/(m + 1)

function square!(out, u)
    at(i, j) = (1 <= i <= m && 1 <= j <= m) ? u[i, j] : zero(u[1, 1])
    for j in 1:m, i in 1:m
        out[i, j] = (at(i - 1, j) + at(i + 1, j) + at(i, j - 1) + at(i, j + 1) - 4u[i, j])/k^2 + exp(u[i, j])
    end
    return out
end


# ---- the search ----
u   = zeros(m, m)
out = nkroot!(square!, u; Δ=1.0, krylov_dim=120, krylov_tol=1e-6, tol=1e-6)

println(out)
println("size of the solution: ", size(u), ", maximum ", maximum(u))
