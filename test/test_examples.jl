# The examples of the README, as tests. Each is a reduced version of an example of examples/, small
# enough to run in seconds, with the convergence of its searches and one invariant of its physics.
module ExamplesTest

using FFTW
using LinearAlgebra
using NKRoots
using Test


# ---------------------------------------------------------------------------- #
# what the tests share: preconditioners, the channel and Kolmogorov flow       #
# ---------------------------------------------------------------------------- #

# the diagonal of the Jacobian of the Broyden function, 3 - 4xᵢ, as preconditioner, with the default
# gram and with a gram of its own
struct Jacobi
    d::Vector{Float64} # the diagonal of the Jacobian at the last point
end
NKRoots.update!(M::Jacobi, x)          = (M.d .= 3 .- 4 .* x; M)
LinearAlgebra.ldiv!(out, M::Jacobi, v) = (out .= v ./ M.d; out)

struct JacobiGram
    d::Vector{Float64} # the diagonal of the Jacobian at the last point
end
NKRoots.update!(M::JacobiGram, x)          = (M.d .= 3 .- 4 .* x; M)
LinearAlgebra.ldiv!(out, M::JacobiGram, v) = (out .= v ./ M.d; out)
NKRoots.gram(M::JacobiGram, Q)             = (w = 1 ./ M.d.^2; [sum(w .* Q[i] .* Q[j]) for i in eachindex(Q), j in eachindex(Q)])

# The example uses fourth-order differences from FDGrids.jl on 300 points. This reduced version has
# the same equations, closures and preconditioner with second-order differences on N points, written
# here, so that the tests need no other package.
const κ   = 0.41
const σ   = 2/3
const cb1 = 0.1355
const cb2 = 0.622
const cw1 = cb1/κ^2 + (1 + cb2)/σ
const cw2 = 0.3
const cw3 = 2.0
const cv1 = 7.1

fv1(χ) = χ^3/(χ^3 + cv1^3)

# the grid, stretched towards the wall, and the matrices of the first and second derivative by three
# points: the value at the wall is zero, and the centre reflects
function channel(N; Reτ=180.0, γ=3.0)
    y  = [1 - tanh(γ*(1 - j/N))/tanh(γ) for j in 0:N]
    D₁ = zeros(N, N)
    D₂ = zeros(N, N)
    for j in 1:N
        h₁    = y[j + 1] - y[j]
        h₂    = j < N ? y[j + 2] - y[j + 1] : h₁
        upper = j < N ? j + 1 : j - 1
        if j > 1
            D₁[j, j - 1] += -h₂/(h₁*(h₁ + h₂))
            D₂[j, j - 1] += 2/(h₁*(h₁ + h₂))
        end
        D₁[j, j]     += (h₂ - h₁)/(h₁*h₂)
        D₂[j, j]     += -2/(h₁*h₂)
        D₁[j, upper] += h₁/(h₂*(h₁ + h₂))
        D₂[j, upper] += 2/(h₂*(h₁ + h₂))
    end

    return (ν = 1/Reτ, N = N, y = y[2:end], D₁ = D₁, D₂ = D₂)
end

# the residual of examples/spalart_allmaras, with x = [U; ν̃]
function channel_residual!(out, x, c)
    N = c.N
    ν = c.ν
    T = eltype(x)
    U = x[1:N]
    n = x[N + 1:2N]

    p   = max.(n, zero(T))
    νt  = p .* fv1.(p ./ ν)
    dU  = c.D₁*U
    d²U = c.D₂*U
    dn  = c.D₁*n
    d²n = c.D₂*n
    dνt = c.D₁*νt

    for i in 1:N
        y  = c.y[i]
        χ  = p[i]/ν
        f₂ = 1 - χ/(1 + χ*fv1(χ))
        Ω  = abs(dU[i])
        S  = max(Ω + p[i]*f₂/(κ*y)^2, 0.3Ω) + 1e-12
        r  = min(p[i]/(S*(κ*y)^2), one(T)*10)
        g  = r + cw2*(r^6 - r)
        fw = g*((1 + cw3^6)/(g^6 + cw3^6))^(1/6)

        out[i]     = (ν + νt[i])*d²U[i] + dνt[i]*dU[i] + 1
        out[N + i] = cb1*S*p[i] - cw1*fw*(p[i]/y)^2 + ((ν + p[i])*d²n[i] + (1 + cb2)*dn[i]^2)/σ
    end

    return out
end

# the preconditioner of the example: the diffusion of each field, factorised again at each point
mutable struct ChannelDiffusion
    c::Any # the channel
    A::Any # LU factorisation of the diffusion of the velocity
    B::Any # and of that of ν̃
end

channel_diffusion(c, a) = Diagonal(a)*c.D₂ + Diagonal(c.D₁*a)*c.D₁

function NKRoots.update!(M::ChannelDiffusion, x)
    c   = M.c
    p   = max.(x[c.N + 1:2c.N], 0.0)
    νt  = p .* fv1.(p ./ c.ν)
    M.A = lu(channel_diffusion(c, c.ν .+ νt))
    M.B = lu(channel_diffusion(c, (c.ν .+ p) ./ σ))
    return M
end

function LinearAlgebra.ldiv!(out, M::ChannelDiffusion, v)
    N = M.c.N
    out[1:N]      .= M.A \ v[1:N]
    out[N + 1:2N] .= M.B \ v[N + 1:2N]
    return out
end

# the search of the example, from its guess unless one is given, and the error of the total stress,
# (ν + νₜ) U' = 1 - y
function channel_search(N; guess=nothing, kwargs...)
    c = channel(N)
    x = guess === nothing ? [@. 20*c.y*(2 - c.y); @. κ*c.y*(1 - c.y) + 3c.ν] : copy(guess)
    t = Trace()
    o = nkroot!((out, v) -> channel_residual!(out, v, c), x; jacobian=:ad, precond=ChannelDiffusion(c, nothing, nothing),
                Δ=1.0, krylov_dim=2N, krylov_tol=1e-3, tol=1e-8, maxiter=100, verbose=false, callback=t, kwargs...)
    U  = x[1:N]
    p  = max.(x[N + 1:2N], 0.0)
    νt = p .* fv1.(p ./ c.ν)

    return (out = o, trace = t, c = c, x = x, U = U, n = x[N + 1:2N],
            stress = maximum(abs, (c.ν .+ νt) .* (c.D₁*U) .- (1 .- c.y)))
end

# the discretisation of the example, as the example includes it, here on 32 × 32 points
module KolmogorovProblem
include(joinpath(@__DIR__, "..", "examples", "kolmogorov", "problem.jl"))
end
const KP = KolmogorovProblem

# N(ω) = u·∇ω - ∇²ω/Re + 4 cos(4y) by the complex transform, with no projection: an evaluation that
# shares nothing with the example
function kolmogorov_direct(ω; Re=40.0)
    N  = size(ω, 1)
    k  = [i <= N÷2 ? i : i - N for i in 0:N - 1]
    kx = [Float64(k[i]) for i in 1:N, j in 1:N]
    ky = [Float64(k[j]) for i in 1:N, j in 1:N]
    k² = kx.^2 .+ ky.^2
    ω̂  = fft(ω)
    ψ̂  = ω̂ ./ max.(k², 1)
    u  = real.(ifft(im .* ky .* ψ̂))
    v  = real.(ifft(-im .* kx .* ψ̂))
    ωx = real.(ifft(im .* kx .* ω̂))
    ωy = real.(ifft(im .* ky .* ω̂))
    Δω = real.(ifft(-k² .* ω̂))

    return u .* ωx .+ v .* ωy .- Δω ./ Re .+ [4cos(4 * 2π*j/N) for i in 1:N, j in 0:N - 1]
end

# the rotation (x, y) → (-x, -y) of a field on the grid
rotate(ω) = [ω[mod1(2 - i, size(ω, 1)), mod1(2 - j, size(ω, 2))] for i in axes(ω, 1), j in axes(ω, 2)]


@testset verbose=true "the examples of the README" begin

# ---------------------------------------------------------------------------- #
# the globalisations on the Bratu problem, examples/globalisation              #
# ---------------------------------------------------------------------------- #

@testset "globalisation: the Bratu problem from a sin(πx)" begin
    n = 200
    h = 1/(n + 1)
    A = Tridiagonal(fill(1/h^2, n - 1), fill(-2/h^2, n), fill(1/h^2, n - 1))

    function bratu!(out, u)
        for i in 1:n
            left   = i == 1 ? zero(u[i]) : u[i - 1]
            right  = i == n ? zero(u[i]) : u[i + 1]
            out[i] = (left - 2u[i] + right)/h^2 + exp(u[i])
        end
        return out
    end

    function search(a; kwargs...)
        u   = [a*sin(π*i*h) for i in 1:n]
        out = nkroot!(bratu!, u; jacobian=:ad, precond=lu(A), Δ=1.0, krylov_tol=1e-6, tol=1e-8, maxiter=100,
                      verbose=false, kwargs...)
        return (; out, u)
    end

    # ---- the hookstep converges from every guess, with every rule ----
    rules = ((trust=:classical,),
             (trust=:scan, scan=:greedy),
             (trust=:scan, scan=:longest),
             (trust=:classical, acceleration=:geodesic),
             (trust=:scan, scan=:longest, acceleration=:geodesic))
    for rule in rules
        @test all(search(a; rule...).out.status === :converged for a in 1:20)
    end

    # ---- classical Newton and backtracking converge from the low guesses and not from a = 10 ----
    for globalisation in (:none, :backtracking)
        @test all(search(a; globalisation=globalisation).out.status === :converged for a in 1:6)
        @test search(10; globalisation=globalisation).out.status !== :converged
    end

    # ---- the two solutions: the lower one from the low guesses, the upper one from the tall ones ----
    @test maximum(search(1).u) ≈ 0.1405 atol=1e-3
    @test maximum(search(10).u) ≈ 4.091 atol=1e-2
end


# ---------------------------------------------------------------------------- #
# a gram of one's own, examples/preconditioner                                 #
# ---------------------------------------------------------------------------- #

@testset "preconditioner: a diagonal preconditioner with a gram of its own" begin
    function broyden!(out, x)
        m = length(x)
        for i in 1:m
            out[i] = (3 - 2x[i])*x[i] + 1 - (i > 1 ? x[i - 1] : 0.0) - 2*(i < m ? x[i + 1] : 0.0)
        end
        return out
    end

    function search(precond)
        t = Trace()
        x = fill(-1.0, 50)
        o = nkroot!(broyden!, x; precond=precond, Δ=0.1, krylov_tol=1e-6, tol=1e-10, maxiter=50, verbose=false, callback=t)
        return o, t
    end
    (o₁, default), (o₂, own) = search(Jacobi(zeros(50))), search(JacobiGram(zeros(50)))

    @test o₁.status === o₂.status === :converged
    @test default.products == own.products
    @test maximum(abs, default.res .- own.res) < 1e-8

    # the gram of its own saves one application of M⁻¹ per Krylov vector
    @test default.preconditioner[end] - own.preconditioner[end] == own.products[end]
end


# ---------------------------------------------------------------------------- #
# the turbulent channel, examples/spalart_allmaras                             #
# ---------------------------------------------------------------------------- #

@testset "the turbulent channel at Reτ = 180, by second differences" begin
    # ---- 120 points from the guess of the example, and 60 points, every second one, from that solution.
    #      From the guess of the example the search on 60 points stalls at a residual of order one ----
    fine   = channel_search(120)
    coarse = channel_search(60; guess=[fine.U[2:2:end]; fine.n[2:2:end]])

    # ---- both searches converge, with a Krylov space far below the size of the system ----
    @test fine.out.status === :converged
    @test coarse.out.status === :converged
    @test maximum(length, fine.trace.krylov) < 60

    # ---- the momentum equation integrates to (ν + νₜ) U' = 1 - y: the error falls at second order ----
    @test fine.stress < 5e-3
    @test 3 < coarse.stress/fine.stress < 5

    # ---- the profile: no slip in the first cell, a positive working variable, a velocity at the
    #      centre of about eighteen friction velocities ----
    @test all(>(0), fine.n)
    @test all(diff(fine.U) .> 0)
    @test 16 < fine.U[end] < 21
    @test isapprox(fine.U[1], 180*fine.c.y[1]; rtol=0.05)

    # ---- without the preconditioner the same search does not converge in the same iterations ----
    @test channel_search(120; precond=nothing, maxiter=fine.out.iter).out.status !== :converged
end

# ---------------------------------------------------------------------------- #
# equilibria of Kolmogorov flow, examples/kolmogorov                           #
# ---------------------------------------------------------------------------- #

@testset "equilibria of Kolmogorov flow, on 32 × 32 points" begin
    K  = KP.Kolmogorov(32; Re=40)
    N  = K.N
    G(ω)  = KP.residual!(similar(ω), ω, K)
    G!(out, ω) = KP.residual!(out, ω, K)
    xs = [2π*i/N for i in 0:N - 1, j in 0:N - 1]
    ys = [2π*j/N for i in 0:N - 1, j in 0:N - 1]

    @testset "the laminar state solves the equations, with dissipation and input Re/32" begin
        ωL = KP.laminar(K)
        b  = KP.budget(K, ωL)
        @test ωL ≈ -10 .* cos.(4 .* ys)
        @test norm(G(ωL)) < 1e-10
        @test b.dissipation ≈ 40/32
        @test b.input ≈ 40/32
        @test b.energy ≈ 25/16
    end

    @testset "the equation keeps the rotation (x, y) → (-x, -y)" begin
        # a smooth field with no symmetry, by the evaluation that does not project
        ω = @. cos(xs + 2ys) + 0.7sin(2xs - ys) + 0.4sin(3xs + ys) + 0.2cos(ys)
        @test norm(ω - rotate(ω)) > 1
        @test kolmogorov_direct(rotate(ω)) ≈ rotate(kolmogorov_direct(ω))

        # the derivative along x of a symmetric field is antisymmetric: the shift leaves the subspace
        s  = @. cos(xs + 2ys) + 0.5cos(3xs - ys)
        sx = @. -sin(xs + 2ys) - 1.5sin(3xs - ys)
        @test s ≈ rotate(s)
        @test sx ≈ -rotate(sx)
    end

    @testset "in the subspace the residual is the equation, and outside it the identity" begin
        # a guess filtered at wavenumber 4: its products stay within the kept band of 10, so that the
        # projector changes nothing and the residual must be N(ω) of the independent evaluation
        ω = KP.guess(K, 1; cutoff=4)
        @test ω ≈ rotate(ω)
        @test abs(sum(ω)) < 1e-10
        @test sum(abs2, ω) / length(ω) ≈ 1
        @test G(ω) ≈ kolmogorov_direct(ω)

        # an antisymmetric field, a mean and a discarded mode come back as they are
        for stray in (sin.(xs .+ ys), fill(0.3, N, N), cos.(13 .* xs))
            @test G(ω .+ stray) .- G(ω) ≈ stray
        end

        # the residual of any field is symmetric but for its stray part
        r = G(ω .+ 0.1 .* sin.(2xs))
        @test r .- rotate(r) ≈ 0.2 .* sin.(2xs)
    end

    @testset "the preconditioner is the Stokes operator, and shortens GMRES" begin
        M = KP.Stokes(K)
        ω = KP.guess(K, 2; cutoff=6)

        # M⁻¹ divides the kept coefficients by |k|²/Re: on cos(3x + 4y) it is a factor 40/25
        c = cos.(3xs .+ 4ys)
        @test ldiv!(similar(c), M, c) ≈ (40/25) .* c
        @test ldiv!(similar(c), M, sin.(xs)) ≈ sin.(xs)

        # the first Newton system of a guess, to a relative residual of 1e-6
        function dimension(precond)
            t = Trace()
            nkroot!(G!, copy(ω); precond=precond, globalisation=:none, krylov_dim=400, krylov_tol=1e-6, maxiter=1,
                    verbose=false, callback=t)
            return length(t.krylov[2])
        end
        @test dimension(M) < dimension(nothing) / 2
    end

    @testset "the searches: back to the laminar state, and to another equilibrium" begin
        M  = KP.Stokes(K)
        ωL = KP.laminar(K)

        # ---- classical Newton from the laminar state with a small symmetric perturbation ----
        t   = Trace()
        ω   = ωL .+ 0.1 .* (cos.(xs .+ ys) .+ cos.(2xs .- 3ys))
        out = nkroot!(G!, ω; precond=M, globalisation=:none, krylov_dim=250, krylov_tol=1e-3, tol=1e-9, maxiter=10,
                      verbose=false, callback=t)
        @test out.status === :converged
        @test norm(ω .- ωL) < 1e-9 * norm(ωL)
        @test all(==(1), t.fraction[2:end])

        # ---- the hookstep from a guess far from it: an equilibrium that is not the laminar state ----
        t   = Trace()
        ω   = 2 .* (cos.(xs) .+ cos.(ys))
        out = nkroot!(G!, ω; precond=M, Δ=5.0, krylov_dim=250, krylov_tol=1e-3, tol=1e-9, maxiter=80, verbose=false,
                      callback=t)
        b   = KP.budget(K, ω)
        @test out.status === :converged
        @test norm(ω .- ωL) > 0.1 * norm(ωL)

        # it is in the subspace: symmetric, of zero mean, with no discarded mode
        @test norm(ω .- rotate(ω)) < 1e-8
        @test abs(sum(ω)) < 1e-8
        @test KP.field(K, KP.project(K, ω)) ≈ ω

        # the energy balance of an equilibrium: the dissipation ⟨ω²⟩/Re is the input ⟨u sin(4y)⟩
        @test isapprox(b.dissipation, b.input; rtol=1e-8)
        @test 0 < b.dissipation < 40/32

        # the matrix of the unknown: every step has the Frobenius norm that the trace records, on the
        # boundary of the trust region or inside it, and every step lowers the residual
        @test ω isa Matrix
        @test all(t.steplength[2:end] .<= t.radius[2:end] .* (1 + 1e-10))
        @test all(diff(t.res) .< 0)
        @test t.preconditioner_updates == t.iter
    end
end

end

end
