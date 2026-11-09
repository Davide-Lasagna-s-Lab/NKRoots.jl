# Every method of NKRoots.jl on every problem of a small set, with every Jacobian. The problems are classical
# tests of Newton's method: a quadratic, the two equations of the README, arctan from far, the Rosenbrock function,
# Powell's badly scaled function, the helical valley of Fletcher and Powell, a narrow valley along a parabola, the
# Broyden tridiagonal function, the Bratu problem with a preconditioner, and a linear system. The methods are the
# hookstep with the classical rule and with the greedy and the longest scans, the classical rule and the longest scan
# with the geodesic acceleration, backtracking and classical Newton; the Jacobians are dual numbers and forward and
# central differences. Every search must converge to a root, but for the failures listed, and every search must keep
# the invariants of its trace.
module MethodsTest

using LinearAlgebra
using NKRoots
using Random
using Test


# ---- the problems ----
quadratic!(out, x) = (out[1] = x[1]^2 - 1; out)

two!(out, x) = (out[1] = x[1]^2 + x[2]^2 - 2; out[2] = exp(x[1] - 1) + x[2]^3 - 2; out)

arctan!(out, x) = (out .= atan.(x); out)

rosenbrock!(out, x) = (out[1] = 10*(x[2] - x[1]^2); out[2] = 1 - x[1]; out)

powell!(out, x) = (out[1] = 1e4*x[1]*x[2] - 1; out[2] = exp(-x[1]) + exp(-x[2]) - 1.0001; out)

# the helical valley, with the angle of Moré, Garbow and Hillstrom, smooth away from x₁ = 0
function helical!(out, x)
    θ      = atan(x[2]/x[1])/(2π) + (x[1] < 0 ? 0.5 : 0.0)
    out[1] = 10*(x[3] - 10θ)
    out[2] = 10*(sqrt(x[1]^2 + x[2]^2) - 1)
    out[3] = x[3]
    return out
end

parabola!(out, x) = (s = x[2] - x[1]^2; out[1] = s; out[2] = s - 1e-3*x[1]; out)

function broyden!(out, x)
    n = length(x)
    for i in 1:n
        out[i] = (3 - 2x[i])*x[i] + 1 - (i > 1 ? x[i - 1] : 0.0) - 2*(i < n ? x[i + 1] : 0.0)
    end
    return out
end

# the Bratu problem by second differences on n points, with its matrix of second differences
function bratu(n)
    h  = 1/(n + 1)
    F! = function (out, u)
        for i in 1:n
            left   = i == 1 ? zero(u[i]) : u[i - 1]
            right  = i == n ? zero(u[i]) : u[i + 1]
            out[i] = (left - 2u[i] + right)/h^2 + exp(u[i])
        end
        return out
    end
    A = Tridiagonal(fill(1/h^2, n - 1), fill(-2/h^2, n), fill(1/h^2, n - 1))

    return F!, A
end

# a linear system A x = c of n unknowns, well conditioned, and its root
function linear(n)
    Random.seed!(3)
    A = Matrix(1.0I, n, n) + 0.3*randn(n, n)/sqrt(n)
    c = randn(n)

    return (out, x) -> (out .= A*x .- c; out), A \ c
end

const BRATU, LAPLACE = bratu(30)
const LINEAR, ROOT   = linear(20)

# name, residual, guess, known roots or nothing, and the keywords of the search
const PROBLEMS = [
    (name = "quadratic",  F! = quadratic!,  x₀ = [2.0],            roots = [[1.0], [-1.0]],   kw = (;)),
    (name = "two",        F! = two!,        x₀ = [2.5, 2.0],       roots = [[1.0, 1.0]],      kw = (;)),
    (name = "arctan",     F! = arctan!,     x₀ = fill(10.0, 3),    roots = [zeros(3)],        kw = (;)),
    (name = "rosenbrock", F! = rosenbrock!, x₀ = [-1.2, 1.0],      roots = [[1.0, 1.0]],      kw = (;)),
    (name = "powell",     F! = powell!,     x₀ = [0.0, 1.0],       roots = nothing,           kw = (;)),
    (name = "helical",    F! = helical!,    x₀ = [-1.0, 0.0, 0.0], roots = [[1.0, 0.0, 0.0]], kw = (;)),
    (name = "parabola",   F! = parabola!,   x₀ = [-1.0, 1.1],      roots = [[0.0, 0.0]],      kw = (;)),
    (name = "broyden",    F! = broyden!,    x₀ = fill(-1.0, 50),   roots = nothing,           kw = (krylov_tol=1e-6,)),
    (name = "bratu",      F! = BRATU,       x₀ = zeros(30),        roots = nothing,
     kw = (precond=lu(LAPLACE), krylov_tol=1e-6)),
    (name = "linear",     F! = LINEAR,      x₀ = zeros(20),        roots = [ROOT],
     kw = (krylov_tol=1e-12, krylov_dim=20)),
]

# name and keywords of every method
const METHODS = [
    (name = "classical",          kw = (trust=:classical,)),
    (name = "greedy",             kw = (trust=:scan, scan=:greedy)),
    (name = "longest",            kw = (trust=:scan, scan=:longest)),
    (name = "classical+geodesic", kw = (trust=:classical, acceleration=:geodesic)),
    (name = "longest+geodesic",   kw = (trust=:scan, scan=:longest, acceleration=:geodesic)),
    (name = "backtracking",       kw = (globalisation=:backtracking,)),
    (name = "newton",             kw = (globalisation=:none,)),
]

# name, keywords and the calls of F per product of every Jacobian
const JACOBIANS = [
    (name = "ad",      kw = (jacobian=:ad,),                       calls = 1),
    (name = "forward", kw = (jacobian=:fd,),                       calls = 1),
    (name = "central", kw = (jacobian=:fd, difference=:central),   calls = 2),
]

# the searches that do not converge in 500 iterations, and why: classical Newton overshoots on arctan from far
# and diverges; backtracking crawls along the narrow bent valley; the acceleration with forward differences on
# Powell's badly scaled function, whose second difference drowns in the error of the first
failure(p, m, j) = (p == "arctan" && m == "newton") ||
                   (p == "parabola" && m == "backtracking") ||
                   (p == "powell" && m == "classical+geodesic" && j == "forward")


@testset "every method on every problem" begin
    for p in PROBLEMS, m in METHODS, j in JACOBIANS
        @testset "$(p.name), $(m.name), $(j.name)" begin
            eta = 1e-3
            t   = Trace()
            x   = copy(p.x₀)
            out = nkroot!(p.F!, x; Δ=1.0, maxiter=500, tol=1e-10, eta=eta, verbose=false, callback=t,
                          p.kw..., m.kw..., j.kw...)

            # ---- converged to a root, but for the failures listed ----
            if failure(p.name, m.name, j.name)
                @test out.status !== :converged
            else
                @test out.status === :converged
                @test out.res < 1e-10
                p.roots === nothing || @test minimum(norm(x .- r) for r in p.roots) < 1e-6
            end

            # ---- the counts: J moves once per step, and F is called by the search and by the Jacobian ----
            @test t.iter == 0:length(t.iter) - 1
            @test t.jacobian_updates == t.iter
            @test t.function_calls == t.residuals .+ j.calls .* t.products
            @test all(issorted, (t.residuals, t.products, t.function_calls))
            if haskey(p.kw, :precond)
                @test t.preconditioner_updates == t.iter
                @test t.grams == t.iter
            end

            # ---- a globalised search accepts only steps that lower the residual ----
            if m.name != "newton"
                @test all(>(eta), t.ratio[2:end])
                @test all(diff(t.res) .< 0)
            end

            # ---- the hookstep stays in its trust region, unless the acceleration lengthens it ----
            if !(m.name in ("backtracking", "newton")) && !occursin("geodesic", m.name)
                @test all(t.steplength[2:end] .<= t.radius[2:end] .* (1 + 1e-10))
            end
        end
    end
end

end
