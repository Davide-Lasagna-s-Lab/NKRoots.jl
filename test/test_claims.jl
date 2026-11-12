# Everything the README says, as a test, section by section. Each testset names what it establishes.

# ---- two equations in two unknowns, with their Jacobian ----
function two!(out, x)
    out[1] = x[1]^2 + x[2]^2 - 2
    out[2] = exp(x[1] - 1) + x[2]^3 - 2
    return out
end

jacobian_two(x) = [2x[1] 2x[2]; exp(x[1] - 1) 3x[2]^2]

struct TwoJacobian
    A::Matrix{Float64} # the matrix at the last point
end

function NKRoots.update!(J::TwoJacobian, x, r)
    J.A .= jacobian_two(x)
    return J
end

LinearAlgebra.mul!(out, J::TwoJacobian, v) = mul!(out, J.A, v)

# a Jacobian that counts its products and its moves, around another
struct CountingJacobian{T}
           J::T                         # the Jacobian
    products::Base.RefValue{Int}        # its products
     updates::Base.RefValue{Int}        # its moves
end
LinearAlgebra.mul!(out, C::CountingJacobian, v) = (C.products[] += 1; mul!(out, C.J, v))
NKRoots.update!(C::CountingJacobian, x, r) = (C.updates[] += 1; NKRoots.update!(C.J, x, r); C)


# ---- the Bratu problem, by second differences on n points ----
function bratu(n)
    h = 1/(n + 1)
    F! = function (out, u)
        for i in 1:n
            left   = i == 1 ? zero(u[i]) : u[i - 1]
            right  = i == n ? zero(u[i]) : u[i + 1]
            out[i] = (left - 2u[i] + right)/h^2 + exp(u[i])
        end
        return out
    end
    A = Tridiagonal(fill(1/h^2, n - 1), fill(-2/h^2, n), fill(1/h^2, n - 1))
    return (; n, h, F!, A)
end

function search(F!, x₀; kwargs...)
    trace = Trace(keep=true)
    x     = copy(x₀)
    out   = nkroot!(F!, x; verbose=false, callback=trace, kwargs...)
    return (; x, out, trace)
end


# ---- a preconditioner that counts its updates, and remembers where it was ----
mutable struct Followed{P}
    M::P                     # the preconditioner
    points::Vector{Vector{Float64}} # the points of its updates
end
Followed(M) = Followed(M, Vector{Float64}[])
NKRoots.update!(F::Followed, x) = (push!(F.points, copy(x)); F)
LinearAlgebra.ldiv!(out, F::Followed, v) = ldiv!(out, F.M, v)

# ---- nonlinear diffusion, (a(u) u')' + 1 = 0 with a = ν + u², and a preconditioner on it ----
function diffusion(n, ν)
    h  = 1/(n + 1)
    a  = (u, i) -> ν + ((i == 0 ? zero(u[1]) : u[i]) + (i == n ? zero(u[1]) : u[i + 1]))^2/4
    F! = function (out, u)
        flux(i) = a(u, i)*((i == n ? zero(u[1]) : u[i + 1]) - (i == 0 ? zero(u[1]) : u[i]))/h
        for i in 1:n
            out[i] = (flux(i) - flux(i - 1))/h + 1
        end
        return out
    end
    operator(u) = (f = [a(u, i)/h^2 for i in 0:n]; Tridiagonal(f[2:n], -(f[1:n] .+ f[2:n + 1]), f[2:n]))
    return (; n, F!, operator)
end

mutable struct Tangent
    operator::Any # the matrix at a point
    F::Any        # its LU factorisation
end
NKRoots.update!(M::Tangent, x) = (M.F = lu(M.operator(x)); M)
LinearAlgebra.ldiv!(out, M::Tangent, v) = ldiv!(out, M.F, v)

# ---- a type of vectors with a basis of its own, the vectors as the columns of one matrix ----
struct Tagged <: AbstractVector{Float64}
    v::Vector{Float64} # the entries
end
Base.size(x::Tagged)                 = size(x.v)
Base.getindex(x::Tagged, i::Int)     = x.v[i]
Base.setindex!(x::Tagged, a, i::Int) = (x.v[i] = a; x)
Base.similar(x::Tagged)              = Tagged(similar(x.v))
Base.copy(x::Tagged)                 = Tagged(copy(x.v))

mutable struct Columns <: AbstractVector{Tagged}
    Q::Matrix{Float64} # the vectors
    n::Int             # how many
    pushed::Base.RefValue{Int} # vectors stored since the start
end
Base.size(B::Columns)            = (B.n,)
Base.getindex(B::Columns, j::Int) = Tagged(B.Q[:, j])
Base.empty!(B::Columns)          = (B.n = 0; B)

const PUSHED = Ref(0)
NKRoots._basis(x::Tagged) = Columns(zeros(length(x), 0), 0, PUSHED)

function NKRoots._push!(B::Columns, v::Tagged)
    B.n += 1
    B.n > size(B.Q, 2) && (B.Q = [B.Q zeros(size(B.Q, 1), 8)])
    B.Q[:, B.n] .= v.v
    B.pushed[] += 1
    return B
end

function NKRoots._orthogonalise!(v::Tagged, B::Columns, h::AbstractVector)
    Q = view(B.Q, :, 1:length(h))
    for _ in 1:2
        c    = Q'*v.v
        v.v .-= Q*c
        h  .+= c
    end
    return v
end

NKRoots.lincomb!(out::Tagged, B::Columns, y::AbstractVector) = (mul!(out.v, view(B.Q, :, 1:length(y)), y); out)


@testset verbose=true "what the README says" begin

@testset "a first search" begin
    @testset "it finds both roots, from different guesses" begin
        a = search(two!, [1.8, 0.7]; Δ=0.1, tol=1e-12)
        b = search(two!, [-1.5, 0.3]; Δ=0.1, tol=1e-12)
        @test a.out.status === :converged
        @test norm(a.x - [1, 1]) < 1e-10
        @test b.out.status === :converged
        @test norm(two!(zeros(2), b.x)) < 1e-11
        @test norm(b.x - [-0.7137, 1.2209]) < 1e-3
    end

    @testset "a small radius doubles while the model is good" begin
        s = search(two!, [1.8, 0.7]; Δ=0.1, tol=1e-12)
        @test s.trace.radius[2:4] ≈ [0.1, 0.2, 0.4]
        @test s.trace.step[2:4] == ["hook", "hook", "hook"]
        @test s.trace.step[end] == "newton"
    end

    @testset "the last steps are quadratic" begin
        s = search(two!, [1.8, 0.7]; jacobian=:ad, Δ=0.1, tol=1e-12)
        r = s.trace.res
        @test r[end - 1] <= 2r[end - 2]^2
        @test r[end] <= 2r[end - 1]^2
    end

    @testset "a guess can end in a valley with no root" begin
        s = search(two!, [1.6, -1.2]; jacobian=:ad, Δ=0.1, maxiter=100)
        J = jacobian_two(s.x)
        r = two!(zeros(2), s.x)
        @test s.out.status === :collapsed
        @test norm(r) > 0.4                # not a root
        @test norm(J'*r) < 1e-5            # but a stationary point of ‖F‖²
    end
end


@testset "the Jacobian" begin
    @testset "finite difference, dual numbers and your own object agree" begin
        ad   = search(two!, [1.8, 0.7]; jacobian=:ad, Δ=0.1, tol=1e-12)
        fd   = search(two!, [1.8, 0.7]; jacobian=:fd, Δ=0.1, tol=1e-12)
        mine = Trace()
        nkroot!(two!, TwoJacobian(zeros(2, 2)), [1.8, 0.7]; Δ=0.1, tol=1e-12, verbose=false, callback=mine)

        # the exact Jacobians agree to rounding, the finite difference until its error shows
        k = length(ad.trace.res) - 1
        @test length(mine.res) == length(ad.trace.res)
        @test maximum(abs, mine.res[1:k] ./ ad.trace.res[1:k] .- 1) < 1e-9
        @test maximum(abs, fd.trace.res[1:5] ./ ad.trace.res[1:5] .- 1) < 1e-6
        @test fd.out.status === :converged
    end

    @testset "ϵ sets the relative step of the finite difference" begin
        x      = [1.8, 0.7]
        exact  = jacobian_two(x)*[1.0, 0.0]
        rough  = jacobianFD(two!, x; ϵ=1e-2)*[1.0, 0.0]
        smooth = jacobianFD(two!, x)*[1.0, 0.0]
        @test 1e-4 < norm(rough - exact) < 1e-1
        @test norm(smooth - exact) < 1e-6

        fine = search(two!, x; Δ=0.1, tol=1e-12)
        bad  = search(two!, x; Δ=0.1, tol=1e-12, ϵ=1e-2)
        @test bad.trace.res != fine.trace.res # the keyword reaches the solver
    end

    @testset "a coarse Jacobian converges, but only linearly" begin
        fine = search(two!, [1.8, 0.7]; jacobian=:ad, Δ=0.1, tol=1e-12)
        bad  = search(two!, [1.8, 0.7]; Δ=0.1, tol=1e-12, ϵ=1e-2, maxiter=40)
        r    = bad.trace.res
        @test bad.out.status === :converged
        @test length(r) > length(fine.trace.res)
        @test r[end]/r[end - 1] > 1e-4 # linear, far from the square of the last residual
    end

    @testset "moving a finite difference costs no evaluation of F" begin
        calls = Ref(0)
        F!    = (o, v) -> (calls[] += 1; two!(o, v))
        s     = search(F!, [1.8, 0.7]; Δ=0.1, tol=1e-12)
        @test calls[] == s.trace.residuals[end] + s.trace.products[end] + 1 # the 1 builds the Jacobian
    end
end


@testset "the Bratu problem" begin
    # the closed form, u(x) = -2 log(cosh((x - 1/2) θ/2)/cosh(θ/4)) with θ = √2 cosh(θ/4)
    closed!(out, θ) = (out[1] = θ[1] - √2*cosh(θ[1]/4); out)
    θ₁ = search(closed!, [1.5]; Δ=0.5, tol=1e-13, maxiter=50).x[1]
    θ₂ = search(closed!, [10.0]; Δ=0.5, tol=1e-13, maxiter=50).x[1]
    exact(p, θ) = [-2log(cosh((i*p.h - 1/2)*θ/2)/cosh(θ/4)) for i in 1:p.n]

    settings = (Δ=1.0, krylov_tol=1e-6, tol=1e-8, maxiter=60)

    @testset "the guess decides which of the two solutions it finds" begin
        p    = bratu(200)
        low  = search(p.F!, zeros(p.n); precond=lu(p.A), settings...)
        high = search(p.F!, [6sin(π*i*p.h) for i in 1:p.n]; precond=lu(p.A), settings...)
        @test low.out.status === :converged
        @test high.out.status === :converged
        @test maximum(low.x) ≈ 0.1405 atol=1e-3
        @test maximum(high.x) ≈ 4.09 atol=0.02
        @test norm(low.x - exact(p, θ₁), Inf) < 1e-6
        @test norm(high.x - exact(p, θ₂), Inf) < 1e-3
    end

    @testset "the error falls as the square of the spacing" begin
        errors = map((100, 200)) do n
            p = bratu(n)
            s = search(p.F!, zeros(n); precond=lu(p.A), settings...)
            norm(s.x - exact(p, θ₁), Inf)
        end
        @test 3.5 < errors[1]/errors[2] < 4.5
    end
end


@testset "the hookstep" begin
    x₀ = [1.8, 0.7]
    r₀ = two!(zeros(2), x₀)
    J₀ = jacobian_two(x₀)

    # one step on the radius Δ, accepted whatever its ratio
    function step(Δ)
        t = Trace(keep=true)
        nkroot!(two!, copy(x₀); jacobian=:ad, Δ=Δ, Δmax=Δ, eta=-Inf, maxiter=1, verbose=false, callback=t)
        return t.x[2] - x₀, t.ratio[2]
    end

    newton = -(J₀ \ r₀)
    ℓ      = norm(newton)

    @testset "the step has the length of the radius, up to the Newton step" begin
        for Δ in (0.01, 0.1, 0.5, 0.9ℓ)
            @test norm(step(Δ)[1]) ≈ Δ rtol=1e-8
        end
        @test step(10ℓ)[1] ≈ newton rtol=1e-8
    end

    @testset "a small radius follows the steepest descent" begin
        δ       = step(1e-5)[1]
        descent = -J₀'*r₀
        @test dot(δ, descent)/(norm(δ)*norm(descent)) > 1 - 1e-6
    end

    @testset "a larger radius lowers the residual of the linear model" begin
        model = [norm(r₀ + J₀*step(Δ)[1]) for Δ in (0.05, 0.1, 0.2, 0.4, 0.8)]
        @test issorted(model; rev=true)
    end

    @testset "the ratio is one for small steps, and the Newton step raises the residual" begin
        @test step(1e-3)[2] ≈ 1 atol=1e-2
        @test step(10ℓ)[2] < 0
        @test norm(two!(zeros(2), x₀ + newton)) > norm(r₀)
    end
end


@testset "the radius" begin
    guesses = ([2.5, 2.0], [1.8, 0.7], [-2.0, 2.0])

    @testset "the rule: by four below 1/4, double above 3/4 on the boundary" begin
        checked = 0
        for guess in guesses, Δ in (0.1, 1.0, 10.0)
            t = search(two!, guess; jacobian=:ad, Δ=Δ, tol=1e-12, maxiter=100).trace
            for k in 2:length(t.res) - 1
                t.rejected[k + 1] == 0 || continue
                expected = t.ratio[k] < 1/4                           ? t.radius[k]/4 :
                           t.ratio[k] > 3/4 && t.step[k] == "hook"    ? 2t.radius[k]  : t.radius[k]
                @test t.radius[k + 1] ≈ expected
                checked += 1
            end
        end
        @test checked > 30
    end

    @testset "a rejected trial always shrinks the radius, whatever eta is" begin
        for η in (0.3, 0.5, 0.9)
            s = search(two!, [3.0, -2.0]; jacobian=:ad, Δ=10.0, eta=η, maxiter=200)
            @test s.out.status in (:collapsed, :converged, :maxiter) # it ends
            for k in 2:length(s.trace.res), (Δ, ρ) in s.trace.trials[k]
                @test !(ρ > η)
            end
        end
    end

    @testset "a rejected step inside the region gives no second residual of the same point" begin
        s = search(arctan!, fill(10.0, 3); Δ=100.0, tol=1e-10, maxiter=50, krylov_tol=1e-8)
        @test s.out.status === :converged
        @test sum(length, s.trace.trials) > 1
        @test all(k -> allunique(last.(s.trace.trials[k])), 2:length(s.trace.res))
    end

    @testset "an infinite radius is the Newton step, and a rejected one comes back finite" begin
        s = search(arctan!, fill(10.0, 3); Δ=Inf, Δmax=Inf, tol=1e-10, maxiter=50, krylov_tol=1e-8)
        @test s.out.status === :converged
        @test s.trace.rejected[2] > 0
        @test all(isfinite, s.trace.radius[3:end])
    end

    @testset "a Jacobian that vanishes ends the search, at any radius" begin
        flat!(out, x) = (out[1] = x[1]^2 + 1; out)
        for Δ in (1.0, Inf), trust in (:classical, :scan)
            @test search(flat!, [0.0]; jacobian=:ad, Δ=Δ, Δmax=Δ, trust=trust).out.status === :collapsed
        end
    end

    @testset "every accepted step has a ratio above eta" begin
        for η in (1e-3, 0.1, 0.5, 0.9)
            t = search(two!, [1.8, 0.7]; jacobian=:ad, Δ=0.9, eta=η, tol=1e-12).trace
            @test all(>(η), t.ratio[2:end])
        end
    end

    @testset "the step never leaves the trust region, and Δmax caps it" begin
        t = search(two!, [2.5, 2.0]; jacobian=:ad, Δ=1.0, Δmax=0.2, tol=1e-12, maxiter=50).trace
        @test all(t.steplength[2:end] .<= t.radius[2:end] .* (1 + 1e-10))
        @test all(t.radius[2:end] .<= 0.2)
    end

    @testset "a small first radius, or a small cap, wastes steps" begin
        steps(; kw...) = search(two!, [2.5, 2.0]; jacobian=:ad, tol=1e-12, maxiter=100, kw...).out.iter
        @test steps(Δ=1e-3) > steps(Δ=1e-1) >= steps(Δ=1.0)
        @test steps(Δ=1.0, Δmax=0.05) > steps(Δ=1.0, Δmax=0.2) > steps(Δ=1.0)
    end

    @testset "trust=:scan finds the same root, keeps the best trial, at no extra product" begin
        ratio = search(two!, [2.5, 2.0]; jacobian=:ad, Δ=1.0, tol=1e-12)
        scan  = search(two!, [2.5, 2.0]; jacobian=:ad, Δ=1.0, tol=1e-12, trust=:scan)
        @test ratio.out.status === :converged
        @test scan.out.status === :converged
        @test norm(ratio.x - scan.x) < 1e-10
        @test scan.trace.res[2] <= ratio.trace.res[2] # the scan holds the trial on Δ itself
        @test scan.trace.residuals[end] > ratio.trace.residuals[end]
        @test diff(scan.trace.products)[1] == diff(ratio.trace.products)[1]
    end

    @testset "scan=:longest finds the same root and keeps the longest step that lowers ‖F‖" begin
        best    = search(two!, [2.5, 2.0]; jacobian=:ad, Δ=1.0, tol=1e-12, trust=:scan)
        longest = search(two!, [2.5, 2.0]; jacobian=:ad, Δ=1.0, tol=1e-12, trust=:scan, scan=:longest)
        @test longest.out.status === :converged
        @test norm(best.x - longest.x) < 1e-10

        # the same trials at the first step, from the same Krylov space: the longest is kept
        @test longest.trace.radius[2] >= best.trace.radius[2]
        @test longest.trace.res[2] >= best.trace.res[2]

        # no trial longer than the accepted one lowers ‖F‖
        t = longest.trace
        for k in 2:length(t.res), (Δ, ρ) in t.trials[k]
            @test Δ < t.radius[k] || !(ρ > 1e-3)
        end
    end

    @testset "scan must be :greedy or :longest" begin
        @test_throws ArgumentError search(two!, [2.5, 2.0]; jacobian=:ad, trust=:scan, scan=:largest)
    end
end


@testset "the longest step" begin
    function valley!(out, x)
        s      = x[2] - x[1]^2
        out[1] = s
        out[2] = s - 1e-3*x[1]
        return out
    end
    run(; kw...) = search(valley!, [-1.0, 1.1]; jacobian=:ad, Δ=1.0, tol=1e-12, maxiter=500, kw...)
    classical = run(trust=:classical)
    longest   = run(trust=:scan, scan=:longest)
    greedy    = run(trust=:scan, scan=:greedy)

    @testset "the root is the origin, and every rule finds it" begin
        for s in (classical, longest, greedy)
            @test s.out.status === :converged
            @test norm(s.x) < 1e-8 # ‖F‖ < 10⁻¹² holds within 10⁻⁹ of the root, along the shear
        end
    end

    @testset "trust=:classical crawls in steps shorter than a hundredth" begin
        @test classical.out.iter > 200
        @test sort(classical.trace.steplength[2:end])[end ÷ 2] < 1e-2
    end

    @testset "scan=:longest takes about a dozen steps, some longer than a tenth" begin
        @test longest.out.iter <= 12
        @test count(>(0.1), longest.trace.steplength[2:end]) >= 2
    end

    @testset "at less than a tenth of the residuals and of the products" begin
        @test longest.trace.residuals[end] < classical.trace.residuals[end] / 10
        @test longest.trace.products[end] < classical.trace.products[end] / 10
    end

    @testset "the longest step is found in a few residuals per step, fewer than the greedy scan" begin
        @test longest.trace.residuals[end] / longest.out.iter < 8
        @test longest.trace.residuals[end] / longest.out.iter < greedy.trace.residuals[end] / greedy.out.iter
    end

    @testset "the greedy scan crawls too, at more than ten times the residuals" begin
        @test greedy.out.iter > 200
        @test greedy.trace.residuals[end] > 10 * classical.trace.residuals[end]
    end
end


@testset "geodesic acceleration" begin
    function valley!(out, x)
        s      = x[2] - x[1]^2
        out[1] = s
        out[2] = s - 1e-3*x[1]
        return out
    end
    run(; kw...) = search(valley!, [-1.0, 1.1]; jacobian=:ad, Δ=1.0, tol=1e-12, maxiter=500, kw...)

    @testset "on the valley the classical rule takes about ten steps, the longest scan fewer than ten" begin
        classical = run(acceleration=:geodesic)
        longest   = run(acceleration=:geodesic, trust=:scan, scan=:longest)
        @test classical.out.status === :converged
        @test longest.out.status === :converged
        @test classical.out.iter <= 12
        @test longest.out.iter < 10
        @test norm(classical.x) < 1e-8
    end

    @testset "an accelerated trial costs two residuals and no product" begin
        plain = run(Δ=0.1, Δmax=0.1, maxiter=1)
        geo   = run(Δ=0.1, Δmax=0.1, maxiter=1, acceleration=:geodesic)
        @test geo.trace.step[2] == "hook"
        @test geo.trace.rejected[2] == 0
        @test diff(geo.trace.residuals) == [2]
        @test diff(geo.trace.products) == diff(plain.trace.products)
    end

    @testset "the GMRES step inside the trust region is taken as it is" begin
        x₀    = [0.0, 1e-3]                  # the GMRES step lands on the root
        plain = search(valley!, x₀; jacobian=:ad, Δ=Inf, maxiter=1)
        geo   = search(valley!, x₀; jacobian=:ad, Δ=Inf, maxiter=1, acceleration=:geodesic)
        @test geo.trace.step[2] == "newton"
        @test diff(geo.trace.residuals) == [1]
        @test geo.x == plain.x
    end

    @testset "a longer acceleration than alpha times the velocity is rejected" begin
        t = run(acceleration=:geodesic, alpha=0.0).trace
        @test all(==("newton"), t.step[2:end])
    end

    @testset "the acceleration needs the hookstep" begin
        @test_throws ArgumentError run(acceleration=:geodesic, globalisation=:backtracking)
        @test_throws ArgumentError run(acceleration=:quadratic)
    end
end

@testset "without a trust region" begin
    arctan!(out, x) = (out .= atan.(x); out)

    @testset "backtracking reaches the root of arctan from 10, where classical Newton diverges" begin
        line = search(arctan!, fill(10.0, 3); jacobian=:ad, globalisation=:backtracking, maxiter=50)
        none = search(arctan!, fill(10.0, 3); jacobian=:ad, globalisation=:none, maxiter=50)
        @test line.out.status === :converged
        @test norm(line.x) < 1e-10
        @test none.out.status !== :converged
    end

    @testset "it halves the GMRES step until the ratio exceeds eta, and calls it a line" begin
        t = search(arctan!, fill(10.0, 3); jacobian=:ad, globalisation=:backtracking, maxiter=50).trace
        @test "line" in t.step
        @test all(>(1e-3), t.ratio[2:end])
        for k in 2:length(t.res)
            lengths = [first.(t.trials[k]); t.radius[k]]
            @test all(isapprox.(lengths[1:end - 1] ./ lengths[2:end], 2; rtol=1e-12))
            @test all(!(ρ > 1e-3) for (_, ρ) in t.trials[k])
        end
    end

    @testset "the step keeps the direction of the Newton step, only shorter" begin
        x₀   = [10.0, -4.0]
        line = search(arctan!, x₀; jacobian=:ad, globalisation=:backtracking, maxiter=1)
        none = search(arctan!, x₀; jacobian=:ad, globalisation=:none, maxiter=1)
        δl   = line.x .- x₀
        δn   = none.x .- x₀
        @test norm(δl) < norm(δn)
        @test abs(dot(δl, δn)) ≈ norm(δl)*norm(δn)
    end

    @testset "classical Newton takes every GMRES step, quadratically near the root" begin
        t = search(two!, [1.8, 0.7]; jacobian=:ad, globalisation=:none, tol=1e-12).trace
        @test all(==(0), t.rejected)
        @test all(==("newton"), t.step[2:end])
        @test t.res[end] < 1e-12
    end

    @testset "Δmax caps the step, and Δ, trust and scan play no part" begin
        t = search(arctan!, fill(10.0, 3); jacobian=:ad, globalisation=:backtracking, Δmax=0.5,
                   maxiter=100).trace
        @test all(t.steplength[2:end] .<= 0.5*(1 + 1e-12))
        a = search(arctan!, fill(10.0, 3); jacobian=:ad, globalisation=:backtracking, maxiter=50)
        b = search(arctan!, fill(10.0, 3); jacobian=:ad, globalisation=:backtracking, maxiter=50,
                   Δ=1e-6, trust=:scan, scan=:longest)
        @test a.x == b.x
        @test a.trace.res == b.trace.res
    end

    @testset "the fraction of the GMRES step is the λ of backtracking, and 1 for a full step" begin
        t = search(arctan!, fill(10.0, 3); jacobian=:ad, globalisation=:backtracking, maxiter=50).trace
        @test isnan(t.fraction[1])
        @test "line" in t.step && "newton" in t.step
        for k in 2:length(t.res)
            @test t.fraction[k] == 2.0^-length(t.trials[k])
            t.step[k] == "newton" && @test t.fraction[k] == 1
        end

        # classical Newton takes the whole step, and a cap of Δmax is part of the fraction
        none = search(two!, [1.8, 0.7]; jacobian=:ad, globalisation=:none, tol=1e-12).trace
        @test all(==(1), none.fraction[2:end])
        cap  = search(arctan!, fill(10.0, 3); jacobian=:ad, globalisation=:backtracking, Δmax=0.5, maxiter=1).trace
        @test cap.fraction[2] < 1
        @test cap.steplength[2] ≈ 0.5 * 2.0^-length(cap.trials[2])
    end

    @testset "the fraction of a hookstep is its length over that of the GMRES step" begin
        x₀ = [1.8, 0.7]
        ℓ  = norm(jacobian_two(x₀) \ two!(zeros(2), x₀))
        t  = search(two!, x₀; jacobian=:ad, Δ=0.1, tol=1e-12).trace
        @test t.step[2] == "hook"
        @test t.fraction[2] ≈ 0.1/ℓ rtol=1e-8
        @test all(t.fraction[k] == 1 for k in 2:length(t.res) if t.step[k] == "newton")
        @test all(t.fraction[k] < 1 for k in 2:length(t.res) if t.step[k] == "hook")
        @test all(==(:hookstep), t.globalisation)
    end

    @testset "the printed table has the same columns for every globalisation, with Δ blank without a trust region" begin
        tables = map((:hookstep, :backtracking, :none)) do globalisation
            io = IOBuffer()
            nkroot!(arctan!, fill(10.0, 3); jacobian=:ad, globalisation=globalisation, maxiter=50, io=io)
            split(String(take!(io)), "\n")
        end
        @test tables[1][2] == tables[2][2] == tables[3][2]
        @test all(occursin(name, tables[1][2]) for name in ("‖δx‖", "Δ", "λ", "ρ"))
        @test any(occursin("↓ line", l) for l in tables[2])

        # the Δ cell, the fourth, is filled in every row with a trust region and blank without
        cell(line) = strip(split(line, "│")[5])
        @test all(!isempty(cell(l)) for l in tables[1][4:end - 3])
        @test all(isempty(cell(l)) for l in tables[2][4:end - 3])
        @test all(isempty(cell(l)) for l in tables[3][4:end - 3])

        # a step that grows without bound keeps the columns aligned, as classical Newton on arctan
        @test length(unique(textwidth.(tables[3][1:end - 2]))) == 1

        # the printed trace has them too
        t     = search(arctan!, fill(10.0, 3); jacobian=:ad, globalisation=:backtracking, maxiter=50).trace
        lines = split(sprint(show, MIME("text/plain"), t), "\n")
        @test all(occursin(name, lines[3]) for name in ("‖δx‖", "Δ", "λ"))
        @test all(isempty(cell(l)) for l in lines[5:end - 1])
    end

    @testset "globalisation must be :hookstep, :backtracking or :none" begin
        @test_throws ArgumentError search(two!, [2.5, 2.0]; jacobian=:ad, globalisation=:linesearch)
    end
end


@testset "the Krylov space" begin
    p = bratu(200)
    M = lu(p.A)
    settings = (precond=M, Δ=1.0, tol=1e-10, maxiter=40)

    @testset "krylov_dim caps the space, krylov_tol stops it" begin
        for d in (1, 2, 5)
            t = search(p.F!, zeros(p.n); krylov_dim=d, krylov_tol=1e-12, settings...).trace
            @test maximum(length, t.krylov) == d
        end
        t = search(p.F!, zeros(p.n); krylov_dim=100, krylov_tol=1e-4, settings...).trace
        @test all(k -> k[end] < 1e-4, t.krylov[2:end])
    end

    @testset "a space of one vector still converges, linearly" begin
        one = search(p.F!, zeros(p.n); krylov_dim=1, settings...)
        all = search(p.F!, zeros(p.n); krylov_tol=1e-8, settings...)
        @test one.out.status === :converged
        @test one.out.iter > 2all.out.iter
    end

    @testset "a tight tolerance gives fewer Newton steps, a loose one fewer products" begin
        loose = search(p.F!, zeros(p.n); krylov_tol=1e-1, settings...).trace
        tight = search(p.F!, zeros(p.n); krylov_tol=1e-8, settings...).trace
        @test length(tight) < length(loose)
        @test tight.products[end] > loose.products[end]
    end

    @testset "the residual of GMRES is the residual of the Newton system" begin
        # with an exact Jacobian: on this stiff operator a finite difference has its own rounding
        s = search(p.F!, zeros(p.n); jacobian=:ad, krylov_tol=1e-6, settings...)
        t = s.trace
        k = findfirst(==("newton"), t.step)  # a full step, that of GMRES
        r = p.F!(zeros(p.n), t.x[k - 1])
        J = Matrix(p.A) + Diagonal(exp.(t.x[k - 1]))
        δ = t.x[k] - t.x[k - 1]
        @test norm(r + J*δ)/norm(r) ≈ t.krylov[k][end] rtol=1e-3
    end
end


@testset "preconditioners" begin
    p = bratu(200)
    settings = (Δ=1.0, krylov_tol=1e-6, tol=1e-8)

    @testset "they change the cost, not the root" begin
        plain = search(p.F!, zeros(p.n); krylov_dim=250, settings...)
        fast  = search(p.F!, zeros(p.n); precond=lu(p.A), settings...)
        @test norm(plain.x - fast.x, Inf) < 1e-10
        @test plain.trace.products[end] > 400
        @test fast.trace.products[end] < 20
    end

    @testset "update!(M, x) follows the search, once per Newton step" begin
        F = Followed(lu(p.A))
        s = search(p.F!, zeros(p.n); precond=F, settings...)
        @test length(F.points) == s.out.iter
        @test F.points == s.trace.x[1:end - 1]
    end

    @testset "the diffusion that follows the point beats the frozen one, and that beats none" begin
        q     = diffusion(60, 1e-2)
        x₀    = fill(1.0, q.n)
        opts  = (Δ=1.0, krylov_tol=1e-6, tol=1e-8, maxiter=100, krylov_dim=120)
        none  = search(q.F!, x₀; opts...).trace.products[end]
        froz  = search(q.F!, x₀; precond=lu(q.operator(x₀)), opts...).trace.products[end]
        moved = search(q.F!, x₀; precond=Tangent(q.operator, nothing), opts...).trace.products[end]
        @test moved < froz < none
    end

    @testset "flexible=true survives a preconditioner that stretches" begin
        Random.seed!(1)
        m      = 60
        Q₀     = Matrix(qr(randn(m, m)).Q)
        B      = Q₀*Diagonal([1e-7; range(1, 3; length=m - 1)])*Q₀'
        c      = randn(m)
        M      = StretchPrecond(Q₀[:, 1], 1e7)
        linear!(out, x) = (mul!(out, B, x); out .-= c; out)

        run(P, flexible) = (t = Trace();
                            nkroot!(linear!, B, zeros(m); precond=P, flexible=flexible, Δ=1e8, krylov_dim=40,
                                    krylov_tol=1e-10, tol=1e-12, maxiter=20, verbose=false, callback=t);
                            t)
        gram     = run(M, false)
        flexible = run(M, true)
        root     = run(StretchRoot(M), false)
        @test flexible.res[2] < 1e-7          # one step
        @test root.res[2] < 1e-7              # gram with a root loses nothing either
        @test minimum(gram.res[1:6]) > 1e-3   # still far after five
        @test minimum(flexible.res) < minimum(gram.res)
    end

    @testset "a type of vectors can bring its own basis, and the search is the same" begin
        PUSHED[] = 0
        q     = bratu(30)
        plain = search(q.F!, zeros(q.n); Δ=1.0, krylov_tol=1e-6, tol=1e-8)
        own   = search(q.F!, Tagged(zeros(q.n)); Δ=1.0, krylov_tol=1e-6, tol=1e-8)
        @test own.out.status === :converged
        @test PUSHED[] >= own.trace.products[end]
        @test own.trace.products == plain.trace.products
        @test abs(own.trace.res[2]/plain.trace.res[2] - 1) < 1e-8
        @test norm(own.x .- plain.x, Inf) < 1e-12
    end
end


@testset "tracing and the output" begin
    p = bratu(100)

    @testset "the trace holds the whole history" begin
        u = [20sin(π*i*p.h) for i in 1:p.n]
        s = search(p.F!, u; precond=lu(p.A), Δ=20.0, krylov_tol=1e-6, tol=1e-9, maxiter=60)
        t = s.trace
        @test length(t) == s.out.iter + 1
        @test t.iter == 0:s.out.iter
        @test t.res[end] == s.out.res
        @test t.x[end] == s.x
        @test length(t.x) == length(t)
        @test all(issorted, (t.products, t.residuals, t.preconditioner, t.time))
        @test t.products[end] == sum(length, t.krylov)
        @test all(==(1), diff(t.iter))
        @test all(in(("hook", "newton")), t.step[2:end])
        @test t.rejected[2:end] == [count(q -> !(q[2] > 1e-3), trials) for trials in t.trials[2:end]]
        @test all(k -> issorted(k; rev=true), t.krylov[2:end])
    end

    @testset "the trace counts every call of F, of the Jacobian and of the preconditioner" begin
        # finite differences, forward and central: one or two calls of F per product, besides the residuals
        for (difference, per) in ((:forward, 1), (:central, 2))
            t = search(two!, [2.5, 2.0]; difference=difference, Δ=1.0, tol=1e-10).trace
            @test t.function_calls == t.residuals .+ per .* t.products
            @test t.jacobian_updates == t.iter
            @test all(==(0), t.preconditioner_updates) && all(==(0), t.grams)
        end

        # dual numbers: one call of F per product
        t = search(two!, [2.5, 2.0]; jacobian=:ad, Δ=1.0, tol=1e-10).trace
        @test t.function_calls == t.residuals .+ t.products

        # a Jacobian of your own: every product and every move is counted, and the calls of F are those of
        # the search
        products = Ref(0)
        updates  = Ref(0)
        J        = TwoJacobian(zeros(2, 2))
        struct_J = CountingJacobian(J, products, updates)
        t        = Trace()
        nkroot!(two!, struct_J, [2.5, 2.0]; Δ=1.0, tol=1e-10, verbose=false, callback=t)
        @test t.products[end] == products[]
        @test t.jacobian_updates[end] == updates[] == t.iter[end]
        @test t.function_calls == t.residuals

        # a preconditioner: moved once per step, with gram once per step unless flexible
        p = bratu(50)
        for flexible in (false, true)
            t = search(p.F!, fill(1.0, p.n); precond=lu(p.A), flexible=flexible, Δ=10.0, tol=1e-8).trace
            @test t.preconditioner_updates == t.iter
            @test t.grams == (flexible ? zero(t.iter) : t.iter)
        end
    end

    @testset "a callback stops the search" begin
        t    = Trace()
        stop = info -> (t(info); info.iter >= 3)
        out  = nkroot!(p.F!, zeros(p.n); precond=lu(p.A), Δ=1.0, maxiter=20, verbose=false, callback=stop)
        @test out.status === :stopped
        @test out.iter == 3
        @test length(t) == 4
    end

    @testset "the four ends of a search" begin
        ok   = nkroot!(two!, [1.8, 0.7]; Δ=0.1, verbose=false)
        many = nkroot!(two!, [1.8, 0.7]; Δ=0.1, maxiter=2, verbose=false)
        val  = nkroot!(two!, [1.6, -1.2]; jacobian=:ad, Δ=0.1, maxiter=100, verbose=false)
        stop = nkroot!(two!, [1.8, 0.7]; Δ=0.1, verbose=false, callback=info -> true)
        @test (ok.status, many.status, val.status, stop.status) == (:converged, :maxiter, :collapsed, :stopped)
    end

    # the lines of a box have one width: the columns are aligned
    aligned(lines) = length(unique(textwidth.(lines))) == 1

    @testset "the printed table is aligned, in every state of the search" begin
        for (guess, kw, mark) in (([1.8, 0.7], (; tol=1e-12), "✓ converged"),
                                  ([1.8, 0.7], (; maxiter=2), "✗ maxiter"),
                                  ([1.6, -1.2], (; jacobian=:ad, maxiter=100), "✗ trust region collapsed"))
            io = IOBuffer()
            nkroot!(two!, guess; Δ=0.1, io=io, kw...)
            text  = String(take!(io))
            lines = split(chomp(text), "\n")
            box   = lines[1:end - 1]
            @test startswith(box[1], "╭") && startswith(box[3], "├") && startswith(box[end], "╰")
            @test aligned(box)
            @test occursin(mark, lines[end])
            @test occursin("‖F‖", box[2]) && occursin("Δ", box[2]) && occursin("ρ", box[2])
            @test occursin("GMRES res.", box[2])
            @test !occursin("NaN", text)
        end

        io = IOBuffer()
        nkroot!(two!, [1.8, 0.7]; Δ=0.1, io=io, callback=info -> info.iter >= 2)
        @test occursin("■ stopped", String(take!(io)))
    end

    @testset "the kinds of step are named" begin
        io = IOBuffer()
        nkroot!(two!, [1.8, 0.7]; Δ=0.1, tol=1e-12, io=io)
        text = String(take!(io))
        @test occursin("↪ hook", text)
        @test occursin("→ newton", text)
    end

    @testset "the printed trace is aligned" begin
        t = search(two!, [1.8, 0.7]; Δ=0.1, tol=1e-12).trace
        lines = split(sprint(show, MIME("text/plain"), t), "\n")
        @test startswith(lines[1], "Trace with $(length(t)) records")
        @test aligned(lines[2:end])
        @test length(lines) == length(t) + 5
    end

    @testset "verbose=false prints nothing" begin
        io = IOBuffer()
        nkroot!(two!, [1.8, 0.7]; Δ=0.1, verbose=false, io=io)
        @test isempty(take!(io))
    end
end


@testset "any type that broadcasts" begin
    # the Bratu problem on a square, with a matrix for the unknown
    n = 20
    h = 1/(n + 1)
    function square!(out, u)
        at(i, j) = (1 <= i <= n && 1 <= j <= n) ? u[i, j] : zero(u[1, 1])
        for j in 1:n, i in 1:n
            out[i, j] = (at(i - 1, j) + at(i + 1, j) + at(i, j - 1) + at(i, j + 1) - 4u[i, j])/h^2 + exp(u[i, j])
        end
        return out
    end
    u   = zeros(n, n)
    out = nkroot!(square!, u; Δ=1.0, krylov_dim=120, krylov_tol=1e-6, tol=1e-6, verbose=false)
    @test out.status === :converged
    @test norm(square!(similar(u), u)) < 1e-6
    @test u ≈ reverse(u; dims=1) rtol=1e-5 # the symmetry of the square
end

end
