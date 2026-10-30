# A turbulent channel with the Spalart–Allmaras model: the streamwise momentum equation and the
# transport equation of the working variable ν̃, solved together.
#
# In units of the half height and of the friction velocity, with ν = 1/Reτ, on 0 ≤ y ≤ 1,
#
#     d/dy [ (ν + νₜ) dU/dy ] + 1 = 0,
#
#     c_b1 S̃ ν̃ - c_w1 f_w (ν̃/y)² + (1/σ) { d/dy [ (ν + ν̃) dν̃/dy ] + c_b2 (dν̃/dy)² } = 0,
#
# with U(0) = ν̃(0) = 0 at the wall and dU/dy = dν̃/dy = 0 at the centre. The closures are
#
#     νₜ = ν̃ f_v1,   f_v1 = χ³/(χ³ + c_v1³),   χ = ν̃/ν,
#     S̃ = |dU/dy| + ν̃ f_v2/(κ y)²,   f_v2 = 1 - χ/(1 + χ f_v1),
#     f_w = g ((1 + c_w3⁶)/(g⁶ + c_w3⁶))^(1/6),   g = r + c_w2 (r⁶ - r),   r = min(ν̃/(S̃ κ² y²), 10),
#
# with S̃ kept above 0.3 |dU/dy|, the usual limiter, and no trip terms.
#
# The stiff part is the diffusion. Finite volumes on a grid stretched towards the wall make it
# tridiagonal in each unknown, and the three preconditioners compared here are
#
#     none,
#     the Laplacian, ν d²/dy² on both unknowns, factorised once,
#     the diffusion, d/dy (ν + νₜ) d/dy and (1/σ) d/dy (ν + ν̃) d/dy, factorised again at each step.
#
# The search goes up in the Reynolds number, each solution the guess of the next. From a parabola the
# search with the diffusion preconditioner converges at Reτ = 180 and stalls at Reτ = 1000. The momentum equation integrates to
# (ν + νₜ) dU/dy = 1 - y, which checks the solution, and in the log layer ν̃ = κ y, as the model was
# built to give.
#
#     julia --project=examples examples/spalart_allmaras.jl
include("common.jl")


# ---- the constants of the model ----
const κ   = 0.41
const σ   = 2/3
const cb1 = 0.1355
const cb2 = 0.622
const cw1 = cb1/κ^2 + (1 + cb2)/σ
const cw2 = 0.3
const cw3 = 2.0
const cv1 = 7.1

fv1(χ) = χ^3/(χ^3 + cv1^3)


# ---- the grid: nodes y[1:N+1], the wall first, unknowns at the nodes 2 to N + 1 ----
struct Channel
      ν::Float64         # the viscosity, 1/Reτ
      N::Int             # the unknowns of each field
      y::Vector{Float64} # nodes
      Δ::Vector{Float64} # distance between the nodes around each face
      w::Vector{Float64} # width of the control volume of each unknown

    function Channel(Reτ::Real, N::Int; γ::Real=3.0)
        ξ    = range(0, 1; length=N + 1)
        y    = @. 1 - tanh(γ*(1 - ξ))/tanh(γ)
        face = (y[1:N] .+ y[2:N + 1]) ./ 2
        return new(1/Reτ, N, y, diff(y), [face[2:N]; 1.0] .- face)
    end
end


# ---- the residual: x = [U; ν̃] ----
function residual!(out, x, c::Channel)
    N = c.N
    ν = c.ν
    T = eltype(x)

    U(i) = i == 0 ? zero(T) : x[i]
    n(i) = i == 0 ? zero(T) : x[N + i]

    # ---- gradients and fluxes at the faces, the face i below the unknown i ----
    gU = [(U(i) - U(i - 1))/c.Δ[i] for i in 1:N]
    gn = [(n(i) - n(i - 1))/c.Δ[i] for i in 1:N]
    nf = [max((n(i) + n(i - 1))/2, zero(T)) for i in 1:N]
    τ  = [(ν + nf[i]*fv1(nf[i]/ν))*gU[i] for i in 1:N]
    q  = [(ν + nf[i])*gn[i]/σ for i in 1:N]

    for i in 1:N
        # ---- the two equations, integrated over the control volume ----
        above = i == N ? zero(T) : τ[i + 1]
        out[i] = above - τ[i] + c.w[i]

        # ---- the gradients at the node, zero above the centre ----
        Ω = abs(gU[i] + (i == N ? zero(T) : gU[i + 1]))/2
        g = (gn[i] + (i == N ? zero(T) : gn[i + 1]))/2

        # ---- the closures ----
        y  = c.y[i + 1]
        p  = max(n(i), zero(T))
        χ  = p/ν
        f₂ = 1 - χ/(1 + χ*fv1(χ))
        S  = max(Ω + p*f₂/(κ*y)^2, 0.3Ω) + 1e-12
        r  = min(p/(S*(κ*y)^2), one(T)*10)
        gw = r + cw2*(r^6 - r)
        fw = gw*((1 + cw3^6)/(gw^6 + cw3^6))^(1/6)

        # ---- production, destruction, diffusion ----
        above = i == N ? zero(T) : q[i + 1]
        out[N + i] = above - q[i] + c.w[i]*(cb1*S*p - cw1*fw*(p/y)^2 + cb2/σ*g^2)
    end

    return out
end


# ---- the preconditioners: one tridiagonal matrix for each field ----
function tridiagonal(a)
    return Tridiagonal(a[2:end], -a .- [a[2:end]; 0.0], a[2:end])
end

mutable struct Diffusion
    c::Channel # the channel
    A::Any     # LU factorisation for the velocity
    B::Any     # and for ν̃

    Diffusion(c::Channel) = new(c, nothing, nothing)
end

function NKRoots.update!(M::Diffusion, x)
    c  = M.c
    N  = c.N
    n  = [0.0; x[N + 1:2N]]
    nf = [max((n[i] + n[i + 1])/2, 0.0) for i in 1:N]

    M.A = lu(tridiagonal(@. (c.ν + nf*fv1(nf/c.ν))/c.Δ))
    M.B = lu(tridiagonal(@. (c.ν + nf)/(σ*c.Δ)))

    return M
end

struct Laplacian
    A::Any # LU factorisation of the second differences times ν, on both fields
end

Laplacian(c::Channel) = Laplacian(lu(tridiagonal(c.ν ./ c.Δ)))

function LinearAlgebra.ldiv!(out, M::Diffusion, v)
    N = M.c.N
    ldiv!(view(out, 1:N), M.A, Vector(view(v, 1:N)))
    ldiv!(view(out, N + 1:2N), M.B, Vector(view(v, N + 1:2N)))
    return out
end

function LinearAlgebra.ldiv!(out, M::Laplacian, v)
    N = length(v) ÷ 2
    ldiv!(view(out, 1:N), M.A, Vector(view(v, 1:N)))
    ldiv!(view(out, N + 1:2N), M.A, Vector(view(v, N + 1:2N)))
    return out
end


# ---- a search from a guess ----
function search(Reτ, x₀, precond; krylov_dim=300, maxiter=100)
    c = Channel(Reτ, 300)
    x = copy(x₀)
    P = precond === :none ? nothing : precond === :laplacian ? Laplacian(c) : Diffusion(c)

    trace = Trace()
    out   = nkroot!((o, v) -> residual!(o, v, c), x; jacobian=:ad, precond=P, Δ=1.0, krylov_dim=krylov_dim,
                    krylov_tol=1e-3, tol=1e-10, maxiter=maxiter, verbose=false, callback=trace)

    @printf "Reτ = %4d, %-9s: %-10s %3d steps, %5d products, ‖F‖ = %.1e\n" Reτ precond out.status out.iter trace.products[end] out.res
    flush(stdout)

    return (; c, x, trace, out)
end


# ---- continuation in the Reynolds number: each solution is the guess of the next ----
c₀      = Channel(180, 300)
y₀      = c₀.y[2:end]
guess   = [@. 20*y₀*(2 - y₀); @. κ*y₀*(1 - y₀) + 3c₀.ν]
numbers = (180, 300, 550, 1000, 2000, 5200)
steps   = map(numbers) do Reτ
    s      = search(Reτ, guess, :diffusion)
    guess .= s.x
    s
end


# ---- the three preconditioners, on the same step of the continuation ----
start = steps[3].x
cases = [search(1000, start, p; maxiter=40) for p in (:none, :laplacian, :diffusion)]


# ---- the profiles, with the check of the stress ----
curves = map(numbers, steps) do Reτ, s
    c = s.c
    N = c.N
    U = s.x[1:N]
    n = s.x[N + 1:2N]

    # ---- the total stress against 1 - y at the faces ----
    Un = [0.0; U]
    nn = [0.0; n]
    nf = (nn[1:N] .+ nn[2:N + 1]) ./ 2
    τ  = @. (c.ν + nf*fv1(nf/c.ν))*(Un[2:N + 1] - Un[1:N])/c.Δ
    y  = (c.y[1:N] .+ c.y[2:N + 1]) ./ 2
    @printf "Reτ = %4d: centre velocity %.3f, stress balance %.1e\n" Reτ U[end] maximum(abs, τ .- (1 .- y))

    (yp=c.y[2:end] .* Reτ, U=U, n=n .* Reτ)
end


# ---- the figure ----
fig    = Figure(size=(900, 380))
shades = ("#86B6E6", "#3F7FC4", "#123F78")
shown  = (1, 4, 6)

ax₁ = Axis(fig[1, 1]; title="the mean velocity", xlabel="y⁺", ylabel="U⁺", xscale=log10, xticks=[1, 10, 100, 1000],
           limits=((0.5, 8000), (0, 30)))
yl = 10 .^ range(log10(20), log10(5000); length=40)
lines!(ax₁, yl, log.(yl) ./ κ .+ 5.0; color=:gray55, linestyle=:dot, linewidth=1.5)
text!(ax₁, 150, 12; text="(1/κ) ln y⁺ + 5", color=INK, fontsize=12)
for (i, shade) in zip(shown, shades)
    lines!(ax₁, curves[i].yp, curves[i].U; color=shade, label="$(numbers[i])")
end
axislegend(ax₁, "Reτ"; position=:lt, labelsize=12, rowgap=0)

ax₂ = Axis(fig[1, 2]; title="the working variable", xlabel="y⁺", ylabel="ν̃⁺", xscale=log10, yscale=log10,
           xticks=[1, 10, 100, 1000], limits=((0.5, 8000), (0.1, 2000)))
lines!(ax₂, yl, κ .* yl; color=:gray55, linestyle=:dot, linewidth=1.5)
text!(ax₂, 25, 40; text="κ y⁺", color=INK, fontsize=12)
for (i, shade) in zip(shown, shades)
    lines!(ax₂, curves[i].yp, max.(curves[i].n, 1e-3); color=shade)
end

ax₃ = Axis(fig[1, 3]; title="three preconditioners, 550 to 1000", xlabel="Jacobian products", ylabel="‖F‖", xscale=log10,
           yscale=log10, yticks=LogTicks(-12:2:0))
for (s, color, name) in zip(cases, SERIES, ("none", "Laplacian", "diffusion"))
    lines!(ax₃, s.trace.products[2:end], s.trace.res[2:end]; color=color, label=name)
end
axislegend(ax₃; position=:lb, labelsize=12, rowgap=0)

finish(fig, "spalart_allmaras")
