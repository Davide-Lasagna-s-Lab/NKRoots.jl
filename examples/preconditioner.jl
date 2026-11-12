# Right preconditioning, in four parts.
#
# 1. A fixed preconditioner. On the Bratu problem the LU factorisation of the second differences cuts
#    the Arnoldi steps of a Newton system from about the size of the system to a few.
#
# 2. A preconditioner that follows the point, by update!(M, x). A nonlinear diffusion,
#    (a(u) u')' + 1 = 0 with a = ν + u², with no preconditioner, with the operator frozen at the
#    guess, and with the operator factorised again at every Newton step.
#
# 3. A gram of your own. A diagonal preconditioner on the Broyden tridiagonal function gives the Gram
#    matrix of the vectors M⁻¹ qⱼ with no application of M⁻¹.
#
# 4. flexible=true, for a preconditioner that solves a few directions exactly. A linear system has
#    one direction with a tiny eigenvalue, and M⁻¹ stretches it by the inverse of that eigenvalue. A
#    Gram matrix of the vectors M⁻¹ qⱼ squares this spread, and the short vectors are lost to
#    rounding. Flexible GMRES keeps them, and so does a gram that returns the triangular root.
#
#     julia --project=examples examples/preconditioner.jl
using Random
include("common.jl")
include("bratu_problem.jl")


# ---------------------------------------------------------------------------- #
# 1. a fixed preconditioner                                                    #
# ---------------------------------------------------------------------------- #

# ---- the Bratu problem from zero, with and without the preconditioner ----
# Settings: n = 200, Δ = 1, krylov_tol = 1e-6, tol = 1e-8, finite-difference Jacobian; krylov_dim = 250
# without the preconditioner, 50 with it.
function bratu_search(; kwargs...)
    trace = Trace()
    u     = zeros(n)
    nkroot!(bratu!, u; Δ=1.0, krylov_tol=1e-6, tol=1e-8, verbose=false, callback=trace, kwargs...)
    return trace
end

plain = bratu_search(krylov_dim=250)
fast  = bratu_search(precond=lu(A))
println("1. the Bratu problem from zero")
@printf "   no preconditioner  %2d steps, %3d Jacobian products\n" plain.iter[end] plain.products[end]
@printf "   precond=lu(A)      %2d steps, %3d Jacobian products\n" fast.iter[end] fast.products[end]


# ---------------------------------------------------------------------------- #
# 2. a preconditioner that follows the point                                   #
# ---------------------------------------------------------------------------- #

# ---- the nonlinear diffusion on N points, with u = 0 at both ends ----
const N = 60
const k = 1/(N + 1)
const ν = 1e-2

# u at the point i, and the diffusivity between the points i and i + 1
at(u, i)          = 1 <= i <= N ? u[i] : zero(u[1])
diffusivity(u, i) = ν + (at(u, i) + at(u, i + 1))^2/4

function diffusion!(out, u)
    for i in 1:N
        right  = diffusivity(u, i)*(at(u, i + 1) - at(u, i))/k
        left   = diffusivity(u, i - 1)*(at(u, i) - at(u, i - 1))/k
        out[i] = (right - left)/k + 1
    end
    return out
end

# the operator with the diffusivity frozen at u, a tridiagonal matrix
function operator(u)
    a = [diffusivity(u, i)/k^2 for i in 0:N]
    return Tridiagonal(a[2:N], -(a[1:N] .+ a[2:N + 1]), a[2:N])
end

# ---- the preconditioner: the operator at the point of the search, factorised by update! ----
mutable struct Diffusion
    F::Any # LU factorisation of the operator at the last point
end

NKRoots.update!(M::Diffusion, x)          = (M.F = lu(operator(x)); M)
LinearAlgebra.ldiv!(out, M::Diffusion, v) = ldiv!(out, M.F, v)

# ---- the three searches ----
# Settings: N = 60, ν = 1e-2, guess u = 1, Δ = 1, krylov_tol = 1e-6, krylov_dim = 120, tol = 1e-8,
# maxiter = 100, finite-difference Jacobian.
function diffusion_search(precond)
    trace = Trace()
    u     = fill(1.0, N)
    out   = nkroot!(diffusion!, u; precond=precond,
                                   Δ=1.0,
                                   krylov_tol=1e-6,
                                   krylov_dim=120,
                                   tol=1e-8,
                                   maxiter=100,
                                   verbose=false,
                                   callback=trace)
    return (; out, trace)
end

none   = diffusion_search(nothing)
frozen = diffusion_search(lu(operator(fill(1.0, N))))
moved  = diffusion_search(Diffusion(nothing))
println("2. the nonlinear diffusion")
for (name, s) in (("no preconditioner", none), ("frozen at the guess", frozen), ("moved by update!", moved))
    t = s.trace
    @printf "   %-20s %-9s %2d steps, %4d Jacobian products\n" name s.out.status s.out.iter t.products[end]
end


# ---------------------------------------------------------------------------- #
# 3. a gram of your own                                                        #
# ---------------------------------------------------------------------------- #

# ---- the Broyden tridiagonal function ----
function broyden!(out, x)
    m = length(x)
    for i in 1:m
        out[i] = (3 - 2x[i])*x[i] + 1 - (i > 1 ? x[i - 1] : 0.0) - 2*(i < m ? x[i + 1] : 0.0)
    end
    return out
end

# ---- the diagonal of its Jacobian, 3 - 4xᵢ, as preconditioner ----
struct Jacobi
    d::Vector{Float64} # the diagonal at the last point
end

NKRoots.update!(M::Jacobi, x)          = (M.d .= 3 .- 4 .* x; M)
LinearAlgebra.ldiv!(out, M::Jacobi, v) = (out .= v ./ M.d; out)

# the same, with the Gram matrix of the vectors qⱼ/d from the weighted inner products of the qⱼ
struct JacobiGram
    d::Vector{Float64} # the diagonal at the last point
end

NKRoots.update!(M::JacobiGram, x)          = (M.d .= 3 .- 4 .* x; M)
LinearAlgebra.ldiv!(out, M::JacobiGram, v) = (out .= v ./ M.d; out)

function NKRoots.gram(M::JacobiGram, Q)
    w = 1 ./ M.d.^2
    return [sum(w .* Q[i] .* Q[j]) for i in eachindex(Q), j in eachindex(Q)]
end

# ---- the two searches ----
# Settings: 50 unknowns, guess x = -1, Δ = 0.1, krylov_tol = 1e-6, tol = 1e-10, maxiter = 50,
# finite-difference Jacobian.
function broyden_search(precond)
    trace = Trace()
    x     = fill(-1.0, 50)
    nkroot!(broyden!, x; precond=precond, Δ=0.1, krylov_tol=1e-6, tol=1e-10, maxiter=50, verbose=false, callback=trace)
    return trace
end

default = broyden_search(Jacobi(zeros(50)))
own     = broyden_search(JacobiGram(zeros(50)))
println("3. the Broyden tridiagonal function with a diagonal preconditioner")
for (name, t) in (("default gram", default), ("gram of your own", own))
    @printf "   %-17s %2d steps, %3d Jacobian products, %3d applications of M⁻¹, %d calls of gram\n" name t.iter[end] t.products[end] t.preconditioner[end] t.grams[end]
end
@printf "   largest difference of the residual histories: %.1e\n" maximum(abs, default.res .- own.res)


# ---------------------------------------------------------------------------- #
# 4. flexible GMRES                                                            #
# ---------------------------------------------------------------------------- #

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

# the same, with a gram that returns the triangular root of the Gram matrix, from a QR of the vectors
struct SlowRoot
    M::Slow # the preconditioner
end

LinearAlgebra.ldiv!(out, M::SlowRoot, q) = ldiv!(out, M.M, q)

function NKRoots.gram(M::SlowRoot, Q)
    Z = reduce(hcat, [ldiv!(similar(q), M, q) for q in Q])
    return UpperTriangular(qr(Z).R)
end

function linear!(out, x)
    mul!(out, B, x)
    out .-= c
    return out
end

# ---- the three searches ----
# Settings: 60 unknowns, eigenvalues 1e-7 and 1 to 3, guess zero, Δ = 1e8, krylov_dim = 40,
# krylov_tol = 1e-10, tol = 1e-12, maxiter = 20, the matrix B as Jacobian.
function solve(precond; flexible=false)
    trace = Trace()
    x     = zeros(m)
    nkroot!(linear!, B, x; precond=precond,
                           flexible=flexible,
                           Δ=1e8,
                           krylov_dim=40,
                           krylov_tol=1e-10,
                           tol=1e-12,
                           maxiter=20,
                           verbose=false,
                           callback=trace)
    return trace
end

matrix = solve(Slow(slow, 1e-7))
root   = solve(SlowRoot(Slow(slow, 1e-7)))
flex   = solve(Slow(slow, 1e-7); flexible=true)
println("4. a preconditioner that stretches one direction by 1e7")
for (name, t) in (("default gram, a Gram matrix", matrix), ("gram returns the root R", root), ("flexible=true", flex))
    @printf "   %-28s ‖F‖ = %.1e after one step, %.1e after %d\n" name t.res[2] t.res[end] t.iter[end]
end


# ---- the figure ----
fig = Figure(size=(1050, 380))

ax₁ = Axis(fig[1, 1]; title="Bratu: GMRES on the first Newton system", xlabel="Arnoldi step",
           ylabel="relative residual", yscale=log10, yticks=LogTicks(-8:2:0), limits=(nothing, (1e-8, 1e0)))
lines!(ax₁, eachindex(plain.krylov[2]), plain.krylov[2]; color=BLUE, label="no preconditioner")
scatterlines!(ax₁, eachindex(fast.krylov[2]), fast.krylov[2]; color=ORANGE, label="precond=lu(A)")
axislegend(ax₁; position=:cc, labelsize=12)

ax₂ = Axis(fig[1, 2]; title="nonlinear diffusion: the cost", xlabel="Jacobian products", ylabel="‖F‖",
           xscale=log10, yscale=log10, xticks=LogTicks(0:4), yticks=LogTicks(-10:2:2),
           limits=((1e0, 1e4), (1e-10, 1e2)))
for (s, color, name) in ((none, BLUE, "no preconditioner"), (frozen, YELLOW, "frozen at the guess"), (moved, ORANGE, "moved by update!"))
    t = s.trace
    scatterlines!(ax₂, t.products[2:end], t.res[2:end]; color=color, markersize=5, label=name)
end
axislegend(ax₂; position=:rt, labelsize=12)

ax₃ = Axis(fig[1, 3]; title="a preconditioner that stretches", xlabel="Newton step", ylabel="‖F‖", yscale=log10,
           yticks=LogTicks(-16:4:4), limits=(nothing, (1e-16, 1e4)))
scatterlines!(ax₃, matrix.iter, max.(matrix.res, 1e-16); color=BLUE, label="default gram")
scatterlines!(ax₃, root.iter, max.(root.res, 1e-16); color=AQUA, label="gram returns R", marker=:diamond)
scatterlines!(ax₃, flex.iter, max.(flex.res, 1e-16); color=ORANGE, label="flexible=true", marker=:rect)
axislegend(ax₃; position=:rt, labelsize=12)

finish(fig, "preconditioner")
