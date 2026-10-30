# A right preconditioner. On the Bratu problem the LU factorisation of the second differences cuts
# the iterations of GMRES from about the size of the system to three.
#
# flexible=true matters for a preconditioner that solves a few directions exactly. Here a linear
# system has one direction with a tiny eigenvalue, and M⁻¹ stretches it by the inverse of that
# eigenvalue. A Gram matrix of the vectors M⁻¹ qⱼ squares this spread, and the short vectors are lost
# to rounding. Flexible GMRES keeps them.
#
#     julia --project=examples examples/preconditioner.jl
using Random
include("common.jl")
include("bratu_problem.jl")


# ---- with and without the preconditioner ----
function search(; kwargs...)
    trace = Trace()
    u     = zeros(n)
    nkroot!(bratu!, u; Δ=1.0, krylov_tol=1e-6, tol=1e-8, verbose=false, callback=trace, kwargs...)
    return trace
end

plain = search(krylov_dim=250)
fast  = search(precond=lu(A))
@printf "Jacobian products: %d without the preconditioner, %d with it\n" plain.products[end] fast.products[end]


# ---- a nearly singular linear problem, and a preconditioner exact on its slow direction ----
Random.seed!(1)
m      = 60
Q₀     = Matrix(qr(randn(m, m)).Q)
values = [1e-7; range(1, 3; length=m - 1)]
B      = Q₀ * Diagonal(values) * Q₀'
c      = randn(m)
slow   = Q₀[:, 1]

struct Slow
    v::Vector{Float64} # the slow direction
    μ::Float64         # its eigenvalue
end

function LinearAlgebra.ldiv!(out, M::Slow, q)
    out .= q .+ (1/M.μ - 1)*dot(M.v, q) .* M.v
    return out
end

function linear!(out, x)
    mul!(out, B, x)
    out .-= c
    return out
end

function solve(flexible)
    trace = Trace()
    x     = zeros(m)
    nkroot!(linear!, B, x; precond=Slow(slow, 1e-7), flexible=flexible, Δ=1e8, krylov_dim=40, krylov_tol=1e-10, tol=1e-12,
            maxiter=20, verbose=false, callback=trace)
    return trace
end

gram     = solve(false)
flexible = solve(true)


# ---- the figure ----
fig = Figure(size=(780, 360))

ax₁ = Axis(fig[1, 1]; title="GMRES on the first Newton system", xlabel="Arnoldi step", ylabel="relative residual",
           yscale=log10, yticks=LogTicks(-8:2:0), limits=(nothing, (1e-7, 2)))
lines!(ax₁, eachindex(plain.krylov[2]), plain.krylov[2]; color=BLUE, label="no preconditioner")
scatterlines!(ax₁, eachindex(fast.krylov[2]), fast.krylov[2]; color=ORANGE, label="precond=lu(A)")
axislegend(ax₁; position=:rc)

ax₂ = Axis(fig[1, 2]; title="a preconditioner that stretches", xlabel="Newton step", ylabel="‖F‖", yscale=log10)
scatterlines!(ax₂, 0:length(gram.res) - 1, max.(gram.res, 1e-16); color=BLUE, label="Gram matrix")
scatterlines!(ax₂, 0:length(flexible.res) - 1, max.(flexible.res, 1e-16); color=ORANGE, label="flexible=true", marker=:rect)
axislegend(ax₂; position=:rt)

finish(fig, "preconditioner")
