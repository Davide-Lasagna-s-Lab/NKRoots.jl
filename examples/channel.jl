# The mean velocity profile of a turbulent channel, from Prandtl's mixing length.
#
# In units of the half height and of the friction velocity, the mean momentum balance is
#
#     d/dy [ (1/Reτ + ℓ² |dU/dy|) dU/dy ] = -1,     U(0) = 0,     dU/dy(1) = 0,
#
# on 0 ≤ y ≤ 1, with the mixing length of Nikuradse in the outer flow and the damping of van Driest
# at the wall,
#
#     ℓ(y) = (0.14 - 0.08 (1 - y)² - 0.06 (1 - y)⁴) (1 - exp(-y Reτ / 26)).
#
# The eddy viscosity depends on the solution, so the problem is nonlinear. Finite volumes on a grid
# stretched towards the wall give one equation per node i, the difference of the total stress τ
# between the two faces of the node plus the driving force:
#
#     τ(i + 1/2) - τ(i - 1/2) + y(i + 1/2) - y(i - 1/2) = 0,     τ = (1/Reτ + ℓ² |g|) g,
#
# with g the velocity gradient at a face and τ = 0 at the centre of the channel. The Jacobian is
# tridiagonal. Its LU factorisation, updated at every Newton step, is a preconditioner that follows
# the point.
#
# Summing the equations from the centre gives τ = 1 - y at every face, a quadratic equation for g.
# The discrete solution is therefore known exactly, and it checks the search.
#
#     julia --project=examples examples/channel.jl
include("common.jl")


# ---- the grid, the mixing length ----
# The unknowns are the velocities at the nodes y[2:N+1], with U = 0 at the wall y[1]. Face i lies
# between the nodes y[i] and y[i + 1], and the control volume of unknown i runs from face i to
# face i + 1, or to the centre of the channel for the last one.
struct Channel
     Reτ::Float64         # friction Reynolds number
       y::Vector{Float64} # nodes, N + 1
       Δ::Vector{Float64} # distance between the nodes around each face
    face::Vector{Float64} # faces, N
       ℓ::Vector{Float64} # mixing length at the faces

    function Channel(Reτ::Real, N::Int; γ::Real=3.0)
        ξ    = range(0, 1; length=N + 1)
        y    = @. 1 - tanh(γ*(1 - ξ))/tanh(γ)
        face = (y[1:N] .+ y[2:N + 1]) ./ 2
        ℓ    = @. (0.14 - 0.08*(1 - face)^2 - 0.06*(1 - face)^4)*(1 - exp(-face*Reτ/26))
        return new(Reτ, y, diff(y), face, ℓ)
    end
end


# ---- the residual ----
function residual!(out, U, c::Channel)
    N = length(U)

    # ---- the stress at the faces ----
    τ = map(1:N) do i
        g = (U[i] - (i == 1 ? zero(U[i]) : U[i - 1]))/c.Δ[i]
        (1/c.Reτ + c.ℓ[i]^2*abs(g))*g
    end

    # ---- the balance of each control volume, with no stress at the centre ----
    for i in 1:N
        above  = i == N ? zero(U[i]) : τ[i + 1]
        top    = i == N ? 1.0 : c.face[i + 1]
        out[i] = above - τ[i] + top - c.face[i]
    end

    return out
end


# ---- the Jacobian as a tridiagonal matrix, its LU factorisation the preconditioner ----
mutable struct Tangent
    c::Channel # the channel
    F::Any     # LU factorisation of the Jacobian at the last point

    Tangent(c::Channel) = new(c, nothing)
end

function NKRoots.update!(M::Tangent, U)
    c = M.c
    N = length(U)

    # ---- dτ/dg over the distance between the nodes, at the faces ----
    a = map(1:N) do i
        g = (U[i] - (i == 1 ? 0.0 : U[i - 1]))/c.Δ[i]
        (1/c.Reτ + 2c.ℓ[i]^2*abs(g))/c.Δ[i]
    end

    # ---- the three diagonals ----
    lower = a[2:N]
    upper = a[2:N]
    diago = -a .- [a[2:N]; 0.0]

    M.F = lu(Tridiagonal(lower, diago, upper))
    return M
end

LinearAlgebra.ldiv!(out, M::Tangent, v) = ldiv!(out, M.F, v)


# ---- the discrete solution from τ = 1 - y at the faces, by the quadratic equation for g ----
function exact(c::Channel)
    r = 1 .- c.face
    g = @. 2r/(1/c.Reτ + sqrt(1/c.Reτ^2 + 4c.ℓ^2*r))
    return cumsum(g .* c.Δ)
end


# ---- the searches ----
N      = 400
cases  = (180, 1000, 5200)
curves = map(cases) do Reτ
    c = Channel(Reτ, N)
    U = @. 20*c.y[2:end]*(2 - c.y[2:end]) # a parabola as a guess
    M = Tangent(c)

    trace = Trace()
    out   = nkroot!((o, u) -> residual!(o, u, c), U; precond=M, Δ=10.0, maxiter=50, tol=1e-11, verbose=false, callback=trace)

    @printf "Reτ = %4d: %s in %d steps, %d Jacobian products, centre velocity %.3f, error against the exact solution %.1e\n" Reτ out.status out.iter trace.products[end] U[end] norm(U - exact(c), Inf)

    (y=c.y[2:end] .* Reτ, U=U)
end


# ---- the profiles in wall units ----
fig = Figure(size=(640, 400))
ax  = Axis(fig[1, 1];
           xscale=log10,
           xlabel="y⁺",
           ylabel="U⁺",
           xticks=[1, 10, 100, 1000],
           limits=((0.5, 12000), (0, 30)))

# ---- the law of the wall and the log law ----
ys = 10 .^ range(log10(0.5), log10(15); length=50)
yl = 10 .^ range(log10(20), log10(6000); length=50)
lines!(ax, ys, ys; color=:gray55, linestyle=:dash, linewidth=1.5)
lines!(ax, yl, log.(yl) ./ 0.4 .+ 5.3; color=:gray55, linestyle=:dot, linewidth=1.5)
text!(ax, 2.2, 7.5; text="U⁺ = y⁺", color=INK, fontsize=13)
text!(ax, 150, 11; text="U⁺ = (1/0.4) ln y⁺ + 5.3", color=INK, fontsize=13)

# ---- one hue, darker with the Reynolds number, each curve named at its end ----
shades = ("#86B6E6", "#3F7FC4", "#123F78")
for (curve, Reτ, shade) in zip(curves, cases, shades)
    lines!(ax, curve.y, curve.U; color=shade)
    text!(ax, curve.y[end], curve.U[end] + 1; text="Reτ = $Reτ", color=INK, fontsize=13, align=(:center, :bottom))
end

finish(fig, "channel")
