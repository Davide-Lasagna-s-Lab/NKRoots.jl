# The preconditioner and the length of the step, checked against dense algebra on small problems.
#
# Every search here runs on a problem of a few unknowns whose Jacobian is known as a matrix, with a
# Krylov space allowed to fill the whole space. The step that the solver takes, the difference of two
# iterates of a trace, is then compared with the same step computed by dense algebra: the Newton
# step -J⁻¹r, the dense trust-region step of a radius, a fraction of the Newton step. The comparison
# does not use the Krylov coordinates, so it tests the metric of the step for every way of giving it:
# no preconditioner, the default gram, a gram that returns the Gram matrix, a gram that returns its
# triangular root, and flexible GMRES.
module PreconditioningTest

using LinearAlgebra
using NKRoots
using Printf
using Random
using Test


# ---------------------------------------------------------------------------- #
# the problems, with their Jacobians as matrices                               #
# ---------------------------------------------------------------------------- #

function broyden!(out, x)
    n = length(x)
    for i in 1:n
        out[i] = (3 - 2x[i])*x[i] + 1 - (i > 1 ? x[i - 1] : 0.0) - 2*(i < n ? x[i + 1] : 0.0)
    end
    return out
end

broyden_jacobian(x) = Matrix(Tridiagonal(fill(-1.0, length(x) - 1), 3 .- 4 .* x, fill(-2.0, length(x) - 1)))

# the Bratu problem by second differences on 8 points
const NB = 8
const HB = 1/(NB + 1)
const AB = Matrix(Tridiagonal(fill(1/HB^2, NB - 1), fill(-2/HB^2, NB), fill(1/HB^2, NB - 1)))

bratu!(out, u)    = (out .= AB*u .+ exp.(u); out)
bratu_jacobian(u) = AB + Diagonal(exp.(u))

two!(out, x)    = (out[1] = x[1]^2 + x[2]^2 - 2; out[2] = exp(x[1] - 1) + x[2]^3 - 2; out)
two_jacobian(x) = [2x[1] 2x[2]; exp(x[1] - 1) 3x[2]^2]

# arctan from far, where the Newton step overshoots: many shortened steps, and classical Newton diverges
arctan!(out, x)    = (out .= atan.(x); out)
arctan_jacobian(x) = Matrix(Diagonal(1 ./ (1 .+ x.^2)))

const PROBLEMS = [(name = "broyden", F! = broyden!, J = broyden_jacobian, x₀ = fill(-1.0, 8)),
                  (name = "bratu",   F! = bratu!,   J = bratu_jacobian,   x₀ = [3sin(π*i*HB) for i in 1:NB]),
                  (name = "two",     F! = two!,     J = two_jacobian,     x₀ = [2.5, 2.0]),
                  (name = "arctan",  F! = arctan!,  J = arctan_jacobian,  x₀ = [10.0, -4.0, 7.0])]

residual(p, x) = p.F!(similar(x), x)


# ---------------------------------------------------------------------------- #
# the preconditioners                                                          #
# ---------------------------------------------------------------------------- #

# a matrix M with the default gram
struct Plain
    F::Any # LU factorisation of M
end
LinearAlgebra.ldiv!(out, P::Plain, v) = ldiv!(out, P.F, v)

# the same, with a gram that returns the Gram matrix, from the matrix M⁻ᵀM⁻¹ and no application of M⁻¹
struct WithG
    F::Any             # LU factorisation of M
    W::Matrix{Float64} # M⁻ᵀM⁻¹
end
LinearAlgebra.ldiv!(out, P::WithG, v) = ldiv!(out, P.F, v)
NKRoots.gram(P::WithG, Q)             = (B = hcat(Q...); B' * P.W * B)

# the same, with a gram that returns the triangular root of the Gram matrix, from a QR of the vectors
struct WithR
    F::Any # LU factorisation of M
end
LinearAlgebra.ldiv!(out, P::WithR, v) = ldiv!(out, P.F, v)
NKRoots.gram(P::WithR, Q)             = UpperTriangular(qr(P.F \ hcat(Q...)).R)

# the ways of giving the metric of the step, from a matrix M: name, preconditioner and flexible
metrics(M) = [(name = "default gram", precond = Plain(lu(M)),                    flexible = false),
              (name = "gram, G",      precond = WithG(lu(M), inv(M)' * inv(M)),  flexible = false),
              (name = "gram, R",      precond = WithR(lu(M)),                    flexible = false),
              (name = "flexible",     precond = Plain(lu(M)),                    flexible = true)]

# two matrices for a problem: a diagonal that scales the directions over three decades, so that the
# Krylov coordinates of a step are far from its length, and the Jacobian at the guess, perturbed
function matrices(p)
    n = length(p.x₀)
    Random.seed!(11)
    scaled = Matrix(Diagonal(10 .^ range(-1.5, 1.5; length=n)))
    near   = p.J(p.x₀) * (I + 0.1 .* randn(n, n))
    return (("scaled", scaled), ("near", near))
end

# every configuration of a problem: none, and the four metrics for each of the two matrices
function configurations(p)
    list = Any[(name = "no preconditioner", precond = nothing, flexible = false)]
    for (label, M) in matrices(p), m in metrics(M)
        push!(list, (name = "$label, $(m.name)", precond = m.precond, flexible = m.flexible))
    end
    return list
end

const METHODS = [(name = "classical",    kw = (trust=:classical,)),
                 (name = "greedy",       kw = (trust=:scan, scan=:greedy)),
                 (name = "longest",      kw = (trust=:scan, scan=:longest)),
                 (name = "geodesic",     kw = (trust=:classical, acceleration=:geodesic)),
                 (name = "backtracking", kw = (globalisation=:backtracking,)),
                 (name = "none",         kw = (globalisation=:none,))]

hookstep(m) = m.name in ("classical", "greedy", "longest", "geodesic")

# a search with dual numbers, a Krylov space that may fill the space and never stops on its residual
function search(p, c, m; kwargs...)
    trace = Trace(keep=true)
    x     = copy(p.x₀)
    out   = nkroot!(p.F!, x; jacobian=:ad,
                             precond=c.precond,
                             flexible=c.flexible,
                             krylov_dim=length(x),
                             krylov_tol=0.0,
                             Δ=0.5,
                             tol=1e-10,
                             maxiter=25,
                             verbose=false,
                             callback=trace,
                             m.kw...,
                             kwargs...)
    return (; x, out, trace)
end


# ---------------------------------------------------------------------------- #
# dense algebra                                                                #
# ---------------------------------------------------------------------------- #

# the trust-region step of the whole space, min ‖r + J v‖ with ‖v‖ ≤ Δ, and its multiplier
function dense_hookstep(J, r, Δ)
    U, d, V = svd(J)
    q       = U' * (-r)
    len(μ)  = norm(q .* d ./ (d.^2 .+ μ))
    len(0.0) <= Δ && return V * (q ./ d), 0.0

    lo = 0.0
    hi = norm(q) * maximum(d) / Δ
    for _ in 1:200
        μ = (lo + hi)/2
        len(μ) > Δ ? (lo = μ) : (hi = μ)
    end

    return V * (q .* d ./ (d.^2 .+ hi)), hi
end

# two vectors agree to the relative tolerance rtol, with the rounding of a difference of iterates
close(a, b; rtol=1e-6, at=1.0) = norm(a - b) <= rtol * max(norm(a), norm(b)) + 1e-12 * (1 + at)


# ---------------------------------------------------------------------------- #
# the preconditioners of single tests                                          #
# ---------------------------------------------------------------------------- #

# the Jacobian at the point as preconditioner, factorised by update!
mutable struct Exact
    J::Any # the Jacobian as a function of the point
    F::Any # its LU factorisation at the last point
end
NKRoots.update!(M::Exact, x)          = (M.F = lu(M.J(x)); M)
LinearAlgebra.ldiv!(out, M::Exact, v) = ldiv!(out, M.F, v)

# a preconditioner that records every call, in order: the point of each update!, each application
mutable struct Recorder
    F::Any           # LU factorisation of M
    log::Vector{Any} # the calls: a copy of the point for update!, :ldiv for an application
end
NKRoots.update!(M::Recorder, x)          = (push!(M.log, copy(x)); M)
LinearAlgebra.ldiv!(out, M::Recorder, v) = (push!(M.log, :ldiv); ldiv!(out, M.F, v))

# the same with a gram of its own, which records its call and applies nothing
struct RecorderGram
    R::Recorder        # the recorder
    W::Matrix{Float64} # M⁻ᵀM⁻¹
end
NKRoots.update!(M::RecorderGram, x)          = (NKRoots.update!(M.R, x); M)
LinearAlgebra.ldiv!(out, M::RecorderGram, v) = ldiv!(out, M.R, v)
NKRoots.gram(M::RecorderGram, Q)             = (push!(M.R.log, :gram); B = hcat(Q...); B' * M.W * B)

# a diagonal preconditioner that is different at every application
mutable struct Changing
    d::Vector{Float64} # the diagonal
    calls::Int         # the applications so far
end
function LinearAlgebra.ldiv!(out, M::Changing, v)
    M.calls += 1
    out .= v ./ (M.d .* (1 .+ 0.5 .* sin.(M.calls .+ eachindex(v))))
    return out
end

# M⁻¹ = I + (10⁸ - 1) s sᵀ, which stretches the direction s, with the default gram
struct Stretch
    s::Vector{Float64} # the stretched direction, of unit length
end
LinearAlgebra.ldiv!(out, M::Stretch, v) = (out .= v .+ (1e8 - 1) * dot(M.s, v) .* M.s; out)

# the same, with a gram that returns the triangular root, from a QR of the vectors
struct StretchR
    M::Stretch # the preconditioner
end
LinearAlgebra.ldiv!(out, M::StretchR, v) = ldiv!(out, M.M, v)
NKRoots.gram(M::StretchR, Q)             = UpperTriangular(qr(hcat([ldiv!(similar(q), M, q) for q in Q]...)).R)

# a preconditioner for an unknown that is a matrix: the Laplacian of the square, on the matrix as a vector
struct OnMatrices
    F::Any # LU factorisation of the Laplacian of the square
end
LinearAlgebra.ldiv!(out::Matrix, M::OnMatrices, v::Matrix) = (out .= reshape(M.F \ vec(v), size(v)); out)


# ---------------------------------------------------------------------------- #
# the tests                                                                    #
# ---------------------------------------------------------------------------- #

@testset verbose=true "preconditioning and the length of the step" begin

@testset "the step taken is the dense step, for every metric and every rule" begin
    compared = Dict(k => 0 for k in ("newton", "line", "hook", "accelerated"))
    for p in PROBLEMS, c in configurations(p), m in METHODS
        @testset "$(p.name), $(c.name), $(m.name)" begin
            s = search(p, c, m)
            t = s.trace
            n = length(p.x₀)

            # ---- the hookstep converges on these problems, but for arctan, whose residual also falls
            #      towards infinity, and every globalised step lowers ‖F‖ ----
            hookstep(m) && p.name != "arctan" && @test s.out.status === :converged
            m.name == "none" || @test all(diff(t.res) .< 0)

            for k in 2:length(t)
                x = t.x[k - 1]
                δ = t.x[k] - x
                ℓ = t.steplength[k]
                Δ = t.radius[k]

                # classical Newton may diverge: its steps are checked while they are finite
                all(isfinite, t.x[k]) && ℓ < 1e100 || break

                # ---- the recorded length is the length of the step taken, in the norm of x ----
                @test abs(norm(δ) - ℓ) <= 1e-7 * ℓ + 1e-12 * (1 + norm(x))

                # ---- a hookstep is as long as its radius, a Newton step is shorter; with the acceleration
                #      the radius bounds the velocity, and the correction is at most alpha/2 of it ----
                if m.name == "geodesic"
                    t.step[k] == "hook" && @test abs(ℓ - Δ) <= (0.75/2) * Δ * (1 + 1e-8)
                elseif hookstep(m)
                    t.step[k] == "hook"   && @test isapprox(ℓ, Δ; rtol=1e-8)
                    t.step[k] == "newton" && @test ℓ <= Δ * (1 + 1e-10)

                    # a step on the boundary is shorter than the Newton step: at the length of the
                    # Newton step a scan takes that step itself, and calls it so
                    t.step[k] == "hook"   && @test t.fraction[k] < 1 - 1e-12
                else
                    @test ℓ == Δ
                end

                # ---- with a Krylov space that fills the space, the step is the dense one ----
                length(t.krylov[k]) == n || continue
                J  = p.J(x)
                r  = residual(p, x)
                δN = -(J \ r)
                cond(J) < 1e8 || continue

                # the fraction is the length of the step over that of the Newton step
                @test isapprox(t.fraction[k], ℓ / norm(δN); rtol=1e-6)

                if t.step[k] == "newton"
                    @test close(δ, δN; at=norm(x))
                    @test t.fraction[k] == 1
                    compared["newton"] += 1
                elseif t.step[k] == "line"
                    @test close(δ, t.fraction[k] .* δN; at=norm(x))
                    @test t.fraction[k] == 2.0^-length(t.trials[k])
                    compared["line"] += 1
                elseif m.name != "geodesic"
                    v, _ = dense_hookstep(J, r, Δ)
                    @test close(δ, v; at=norm(x))
                    compared["hook"] += 1
                else
                    # the velocity is the dense hookstep, and the rest of the step is half the
                    # acceleration: the dense solution of (JᵀJ + μ I) a = -Jᵀ F''[v, v], with the
                    # second derivative by the difference of the solver, and no longer than alpha ‖v‖
                    v, μ = dense_hookstep(J, r, Δ)
                    a    = 2 .* (δ .- v)
                    h    = 0.1
                    F2   = (2/h) .* ((residual(p, x .+ h .* v) .- r) ./ h .- J * v)
                    @test norm(a) <= 0.75 * norm(v) * (1 + 1e-6)
                    @test close(a, -((J'J + μ*I) \ (J' * F2)); rtol=1e-5, at=norm(x))
                    compared["accelerated"] += 1
                end
            end
        end
    end

    # every kind of step was compared many times
    @test all(>(20), values(compared))
end

@testset "no step exceeds Δmax, for every rule" begin
    Δmax = 0.2
    for p in PROBLEMS, c in configurations(p), m in METHODS
        m.name == "none" && continue
        t = search(p, c, m; Δmax=Δmax, maxiter=40).trace
        if m.name == "geodesic"
            @test all(t.radius[2:end] .<= Δmax)
            @test all(t.steplength[2:end] .<= (1 + 0.75/2) * Δmax * (1 + 1e-8))
        else
            @test all(t.radius[2:end] .<= Δmax * (1 + 1e-10))
            @test all(t.steplength[2:end] .<= Δmax * (1 + 1e-10))
        end
        @test all(diff(t.res) .< 0)
    end
end

@testset "the residual of GMRES is that of the linear system of the step" begin
    # a Krylov space cut at three vectors, and every step taken as it is: the reported residual is
    # ‖r + J δx‖/‖r‖ with the dense Jacobian, for every metric
    checked = 0
    for p in PROBLEMS[1:2], c in configurations(p)
        t = search(p, c, METHODS[end]; krylov_dim=3, krylov_tol=1e-2, maxiter=6).trace
        for k in 2:length(t)
            x = t.x[k - 1]
            r = residual(p, x)
            δ = t.x[k] - x
            all(isfinite, t.x[k]) || break
            @test isapprox(norm(r + p.J(x) * δ) / norm(r), t.krylov[k][end]; rtol=1e-6, atol=1e-12)
            @test length(t.krylov[k]) <= 3
            checked += 1
        end
    end
    @test checked > 60
end

@testset "with the exact Jacobian as preconditioner the Krylov space has one vector" begin
    for p in PROBLEMS, flexible in (false, true)
        c = (name = "exact", precond = Exact(p.J, nothing), flexible = flexible)
        t = search(p, c, METHODS[1]; krylov_tol=1e-8).trace
        @test all(==(1), length.(t.krylov[2:end]))
        @test all(k -> k[end] < 1e-10, t.krylov[2:end])
        @test t.res[end] < 1e-10
    end
end

@testset "update!(M, x) is called once per iteration, at the point, before the Krylov space" begin
    p = PROBLEMS[1]
    M = matrices(p)[2][2]
    for m in METHODS[1:5], kind in (:default, :own, :flexible)
        rec = Recorder(lu(M), Any[])
        c   = kind === :own ? (name = "", precond = RecorderGram(rec, inv(M)' * inv(M)), flexible = false) :
                              (name = "", precond = rec, flexible = kind === :flexible)
        s   = search(p, c, m; krylov_tol=1e-6)
        t   = s.trace
        log = rec.log

        # ---- one update per iteration, each at the point the iteration starts from ----
        updates = findall(e -> e isa Vector, log)
        @test length(updates) == s.out.iter
        @test log[updates] == t.x[1:end - 1]

        # ---- the first call of every iteration is the update: nothing is applied before it ----
        @test updates[1] == 1
        @test all(log[i + 1] === :ldiv for i in updates)

        # ---- the applications between two updates are those the trace counts for that iteration, and
        #      so are the totals; the default gram applies M⁻¹ once per Krylov vector ----
        bounds = [updates; length(log) + 1]
        seen   = [count(==(:ldiv), log[bounds[i]:bounds[i + 1] - 1]) for i in eachindex(updates)]
        @test seen == diff(t.preconditioner)
        @test count(==(:ldiv), log) == t.preconditioner[end]
        @test t.preconditioner_updates == t.iter

        # ---- gram is called once per iteration, unless flexible; a gram of one's own is seen ----
        @test t.grams == (kind === :flexible ? zero(t.iter) : t.iter)
        kind === :own      && @test count(==(:gram), log) == t.grams[end]
        kind === :flexible && @test t.preconditioner == t.products
        # one application per Krylov vector in the Arnoldi iteration, one in the default gram, and
        # one for the step of every residual after the first
        kind === :default  && @test t.preconditioner[end] == 2t.products[end] + t.residuals[end] - 1
    end
end

@testset "flexible GMRES with a preconditioner that changes between Arnoldi steps" begin
    checked = 0
    for p in PROBLEMS[1:2]
        M = Changing(abs.(diag(p.J(p.x₀))), 0)
        c = (name = "changing", precond = M, flexible = true)
        t = search(p, c, METHODS[end]; krylov_dim=4, krylov_tol=1e-8, maxiter=6).trace
        for k in 2:length(t)
            x = t.x[k - 1]
            r = residual(p, x)
            δ = t.x[k] - x

            # the step solves the linear model to the residual that GMRES reports, and has the recorded length
            @test isapprox(norm(r + p.J(x) * δ) / norm(r), t.krylov[k][end]; rtol=1e-6, atol=1e-12)
            @test isapprox(norm(δ), t.steplength[k]; rtol=1e-8)
            checked += 1
        end
        @test M.calls == t.products[end]
    end
    @test checked >= 10
end

@testset "the default gram, a gram that returns G and one that returns R give the same iterates" begin
    for p in PROBLEMS[1:3], (label, M) in matrices(p), m in METHODS[1:5]
        runs = [search(p, c, m).trace for c in metrics(M)]
        for t in runs[2:end]
            @test length(t) == length(runs[1])
            @test t.step == runs[1].step
            @test all(close(a, b; rtol=1e-7) for (a, b) in zip(t.x, runs[1].x))
        end
    end
end

@testset "the iterates do not change when the preconditioner is multiplied by a constant" begin
    for p in PROBLEMS[1:3], (label, M) in matrices(p), m in METHODS[1:5], f in metrics(M)[[1, 4]]
        a = search(p, f, m).trace
        g = f.flexible ? metrics(1e3 .* M)[4] : metrics(1e3 .* M)[1]
        b = search(p, g, m).trace
        @test length(a) == length(b)
        @test all(close(u, v; rtol=1e-8) for (u, v) in zip(a.x, b.x))
        @test isapprox(a.steplength[2:end], b.steplength[2:end]; rtol=1e-8)
    end
end

@testset "a preconditioner that stretches one direction: the Gram matrix loses the step, R and flexible do not" begin
    # a linear problem A x = c, and M⁻¹ = I + (10⁸ - 1) s sᵀ: the reported residual of GMRES is small
    # in every case, since it does not depend on the metric, and the step solves the system only if
    # the metric keeps the short vectors, as the README documents
    Random.seed!(5)
    n = 30
    A = Matrix(1.0I, n, n) + 0.3 * randn(n, n) / sqrt(n)
    c = randn(n)
    F!(out, x) = (out .= A*x .- c; out)
    M = Stretch(normalize(randn(n)))

    function one(precond, flexible)
        t = Trace(keep=true)
        nkroot!(F!, A, zeros(n); precond=precond, flexible=flexible, globalisation=:none, krylov_dim=n,
                krylov_tol=1e-10, maxiter=1, verbose=false, callback=t)
        return t
    end
    gram = one(M, false)
    root = one(StretchR(M), false)
    flex = one(M, true)

    @test all(t.krylov[2][end] < 1e-10 for t in (gram, root, flex))
    @test root.res[2] < 1e-6 * root.res[1]
    @test flex.res[2] < 1e-6 * flex.res[1]
    @test gram.res[2] > 1e-2 * gram.res[1]

    # the recorded length is that of the step taken: to rounding with R and with flexible GMRES, and
    # to the digits that the Gram matrix keeps of a stretch of 10⁸
    @test isapprox(norm(root.x[2]), root.steplength[2]; rtol=1e-7)
    @test isapprox(norm(flex.x[2]), flex.steplength[2]; rtol=1e-7)
    @test isapprox(norm(gram.x[2]), gram.steplength[2]; rtol=1e-3)
end

@testset "backtracking: halves, the fraction, the cap of Δmax and the collapse" begin
    p = PROBLEMS[2]
    for c in configurations(p)[[1, 2, 4, 5]]
        # from a taller guess the first Newton steps are refused
        t = search(p, c, METHODS[5]; krylov_tol=0.0).trace
        @test "line" in t.step
        for k in 2:length(t)
            # each trial is half the one before, and the accepted one half the last refused
            lengths = [first.(t.trials[k]); t.steplength[k]]
            @test all(isapprox.(lengths[1:end - 1] ./ lengths[2:end], 2; rtol=1e-10))
            @test all(!(ρ > 1e-3) for (_, ρ) in t.trials[k])
            @test t.ratio[k] > 1e-3
        end

        # with a cap the first trial has the length Δmax, and the cap is part of the fraction
        Δmax = 0.3
        t    = search(p, c, METHODS[5]; krylov_tol=0.0, Δmax=Δmax, maxiter=40).trace
        for k in 2:length(t)
            x  = t.x[k - 1]
            ℓN = norm(p.J(x) \ residual(p, x))
            length(t.krylov[k]) == length(x) || continue
            @test isapprox(t.fraction[k], min(1, Δmax/ℓN) * 2.0^-length(t.trials[k]); rtol=1e-6)
            @test isapprox(t.steplength[k], t.fraction[k] * ℓN; rtol=1e-6)
        end
    end

    # a Jacobian that vanishes: the GMRES step is not a number, and the search ends
    flat!(out, x) = (out[1] = x[1]^2 + 1; out)
    for x₀ in ([0.0], [1.0]), Δmax in (Inf, 1.0)
        out = nkroot!(flat!, copy(x₀); jacobian=:ad, globalisation=:backtracking, Δmax=Δmax, verbose=false)
        @test out.status === :collapsed
    end

    # a valley with no root: the steps shrink to the rounding of the point, and the search collapses
    t   = Trace()
    x   = [3.0, -2.0]
    out = nkroot!(two!, x; jacobian=:ad, globalisation=:backtracking, maxiter=100, verbose=false, callback=t)
    @test out.status === :collapsed
    @test out.res > 0.1
    @test out.res == t.res[end]
end

@testset "the numbers of the printed table are those of the trace" begin
    cells(line) = strip.(split(line, "│")[2:end - 1])
    p = PROBLEMS[2]
    for m in METHODS, c in configurations(p)[[1, 2]]
        io = IOBuffer()
        t  = Trace()
        x  = copy(p.x₀)
        nkroot!(p.F!, x; jacobian=:ad, precond=c.precond, Δ=0.5, maxiter=25, io=io, callback=t, m.kw...)
        rows = cells.(split(String(take!(io)), "\n")[4:end - 3])

        # ---- one row per record, with the length, the fraction and the residual of the record ----
        @test length(rows) == length(t)
        for k in 2:length(t)
            @test rows[k][2] == @sprintf("%.4e", t.res[k])
            @test rows[k][3] == @sprintf("%.2e", t.steplength[k])
            @test rows[k][5] == @sprintf("%.2e", t.fraction[k])
            @test rows[k][6] == @sprintf("%.3f", t.ratio[k])
        end

        # ---- Δ is blank without a trust region; with one it is the radius of the next iteration, the
        #      one of its accepted step when the classical rule accepts the first trial ----
        if !hookstep(m)
            @test all(isempty(r[4]) for r in rows)
        else
            @test rows[1][4] == @sprintf("%.2e", t.radius[1])
            m.name == "classical" &&
                @test all(rows[k][4] == @sprintf("%.2e", t.radius[k + 1]) for k in 1:length(t) - 1 if t.rejected[k + 1] == 0)
        end

        # ---- the printed trace: the length, the radius and the fraction of each record ----
        shown = cells.(split(sprint(show, MIME("text/plain"), t), "\n")[5:end - 1])
        @test length(shown) == length(t)
        for k in 2:length(t)
            @test shown[k][3] == @sprintf("%.2e", t.steplength[k])
            @test shown[k][4] == (hookstep(m) ? @sprintf("%.2e", t.radius[k]) : "")
            @test shown[k][5] == @sprintf("%.2e", t.fraction[k])
        end
    end
end

@testset "a matrix for the unknown, with a preconditioner" begin
    # the Bratu problem on a square of 8 × 8 points, with the unknown as a matrix and the Laplacian of
    # the square, factorised, as preconditioner; the same search with the unknown as a vector
    m  = 8
    h  = 1/(m + 1)
    T  = Matrix(Tridiagonal(fill(1/h^2, m - 1), fill(-2/h^2, m), fill(1/h^2, m - 1)))
    L  = kron(Matrix(1.0I, m, m), T) + kron(T, Matrix(1.0I, m, m))
    LU = lu(L)

    square!(out, u) = (out .= T*u .+ u*T .+ exp.(u); out)
    flat!(out, u)   = (out .= L*u .+ exp.(u); out)

    guess = [4sin(π*i*h)*sin(π*j*h) for i in 1:m, j in 1:m]
    for flexible in (false, true), kw in ((;), (trust=:scan, scan=:longest), (globalisation=:backtracking,))
        a  = Trace(keep=true)
        b  = Trace(keep=true)
        u  = copy(guess)
        v  = vec(copy(guess))
        oa = nkroot!(square!, u; jacobian=:ad, precond=OnMatrices(LU), flexible=flexible, Δ=1.0, krylov_tol=1e-8,
                     maxiter=40, verbose=false, callback=a, kw...)
        ob = nkroot!(flat!, v; jacobian=:ad, precond=LU, flexible=flexible, Δ=1.0, krylov_tol=1e-8,
                     maxiter=40, verbose=false, callback=b, kw...)
        @test oa.status === :converged
        @test u isa Matrix
        @test length(a) == length(b)
        @test all(close(vec(x), y; rtol=1e-6) for (x, y) in zip(a.x, b.x)) # GMRES stops at 1e-8

        # the length of the step is the Frobenius norm of the difference of two iterates
        for k in 2:length(a)
            @test isapprox(norm(a.x[k] - a.x[k - 1]), a.steplength[k]; rtol=1e-7, atol=1e-12)
        end
    end
end

end

end
