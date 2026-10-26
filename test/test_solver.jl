# simple one-dimensional function
function test1(out::Vector, x::Vector)
    out[1] = x[1]^2 - 1
    return out
end

# the Broyden tridiagonal function, a classic test of Newton methods, with root near x = -1/2
function broyden!(out::Vector, x::Vector)
    n = length(x)
    for i in 1:n
        out[i] = (3 - 2x[i])*x[i] + 1 - (i > 1 ? x[i-1] : 0.0) - 2*(i < n ? x[i+1] : 0.0)
    end
    return out
end

# a preconditioner with its own Gram matrix, to compare with the default
struct DiagonalPrecond
    d::Vector{Float64}
end
LinearAlgebra.ldiv!(out, M::DiagonalPrecond, v) = (out .= v ./ M.d; out)
NKRoots.gram(M::DiagonalPrecond, Q) = [dot(Q[i] ./ M.d, Q[j] ./ M.d) for i in eachindex(Q), j in eachindex(Q)]

@testset "test 1                                   " begin
    for jacobian in (:ad, :fd), trust in (:ratio, :scan)
        x    = Float64[2.0]
        data = Float64[]
        out  = nkroot!(test1, x; jacobian=jacobian, trust=trust, Δ=1.0, verbose=false,
                       callback=info -> (push!(data, info.x[1]); false))
        @test out.status == :converged
        @test data[1] == 2.0
        @test abs(x[1] - 1.0) < 1e-10
    end
end

@testset "Broyden tridiagonal                      " begin
    for trust in (:ratio, :scan), precond in (nothing, DiagonalPrecond(fill(3.0, 50)))
        x   = fill(-1.0, 50)
        out = nkroot!(broyden!, x; trust=trust, precond=precond, Δ=0.1, maxiter=50, krylov_tol=1e-6,
                      tol=1e-10, verbose=false)
        @test out.status == :converged
        @test norm(broyden!(similar(x), x)) < 1e-10
    end
end

@testset "preconditioned trust region              " begin
    # a linear problem A x = c: ρ = 1 for every step, which the trust region therefore bounds
    Random.seed!(1)
    n = 30
    A = Matrix(1.0I, n, n) + 0.3*randn(n, n)
    c = randn(n)
    F!(out, x) = (out .= A*x .- c; out)
    Δ = 1e-2

    # one step from zero with M = diag(A): its length is Δ, the same with the vectors M⁻¹ qⱼ kept
    # (flexible), with the default Gram matrix and with the one of the preconditioner
    M     = DiagonalPrecond(diag(A))
    steps = Vector{Float64}[]
    for (flexible, P) in ((true, M), (false, M), (false, Diagonal(diag(A))))
        x = zeros(n)
        nkroot!(F!, A, x; precond=P, flexible=flexible, Δ=Δ, Δmax=Δ, maxiter=1, krylov_tol=1e-12,
                krylov_dim=n, verbose=false)
        @test abs(norm(x) - Δ) < 1e-10*Δ
        push!(steps, x)
    end
    @test norm(steps[1] - steps[2]) < 1e-10*Δ
    @test norm(steps[1] - steps[3]) < 1e-10*Δ

    # with M = A the Krylov space has one vector and the Newton step solves the problem
    x   = zeros(n)
    k   = Int[]
    out = nkroot!(F!, A, x; precond=lu(A), Δ=1e3, maxiter=2, verbose=false,
                  callback=info -> (info.iter > 0 && push!(k, length(info.krylov)); false))
    @test out.status == :converged
    @test k[1] == 1
end

# the Jacobian of Broyden by finite differences, counting its products
struct CountedJacobian{T}
          J::T        # finite-difference Jacobian
    products::Base.RefValue{Int}
end
LinearAlgebra.mul!(out, C::CountedJacobian, v) = (C.products[] += 1; mul!(out, C.J, v))
NKRoots.update!(C::CountedJacobian, x, r) = (update!(C.J, x, r); C)

# arctan, on which Newton overshoots from |x| > 1.39
arctan!(out, x) = (out .= atan.(x); out)

@testset "rejected steps                           " begin
    # a large initial radius from x = 10: the Newton steps overshoot and are rejected, and the
    # shorter hooksteps are computed in the same Krylov space, with no product with the Jacobian
    x    = fill(10.0, 3)
    C    = CountedJacobian(jacobianFD(arctan!, x), Ref(0))
    dims = Int[]
    rej  = Int[]
    out  = nkroot!(arctan!, C, x; Δ=100.0, maxiter=50, krylov_tol=1e-8, verbose=false,
                   callback=info -> (info.iter > 0 && (push!(dims, length(info.krylov)); push!(rej, info.rejected)); false))
    @test out.status == :converged
    @test sum(rej) > 0
    @test C.products[] == sum(dims)
end

@testset "trace                                    " begin
    # the trace follows the search: one record at the start and one per iteration, and its counts
    # agree with the products of the counted Jacobian
    x     = fill(10.0, 3)
    C     = CountedJacobian(jacobianFD(arctan!, x), Ref(0))
    trace = Trace(keep=true)
    out   = nkroot!(arctan!, C, x; Δ=100.0, maxiter=50, trust=:scan, verbose=false, callback=trace)
    @test out.status == :converged
    @test length(trace) == out.iter + 1
    @test trace.iter == 0:out.iter
    @test trace.products[end] == C.products[]
    @test trace.res[end] == out.res
    @test trace.x[end] == x
    @test all(issorted, (trace.products, trace.residuals, trace.time))
end
