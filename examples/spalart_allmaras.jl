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
# The derivatives are the fourth-order finite differences of FDGrids.jl, five-point stencils on a
# grid stretched towards the wall, with the even symmetry of both fields at the centre.
#
# The stiff part is the diffusion, 𝒟 = d/dy a d/dy with a = ν + νₜ for the velocity and a = (ν + ν̃)/σ
# for ν̃. As a matrix it is Diagonal(a) D₂ + Diagonal(D₁ a) D₁, on each field. The three
# preconditioners compared here are
#
#     none,
#     the Laplacian, ν D₂ on both fields, factorised once,
#     the diffusion, factorised again at each Newton step.
#
# The search goes up in the Reynolds number, each solution the guess of the next. The momentum equation
# integrates to (ν + νₜ) dU/dy = 1 - y, which checks the solution and its order of accuracy, and in the
# log layer ν̃ = κ y, as the model was built to give.
#
#     julia --project=examples examples/spalart_allmaras.jl
using FDGrids
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


# ---- the grid and the matrices of the derivatives: unknowns at y[2:N+1], the wall is y[1] ----
struct Channel
     ν::Float64         # the viscosity, 1/Reτ
     N::Int             # the unknowns of each field
     y::Vector{Float64} # the nodes of the unknowns
    D₁::Matrix{Float64} # first derivative
    D₂::Matrix{Float64} # second derivative

    function Channel(Reτ::Real, N::Int; γ::Real=3.0)
        ξ    = range(0, 1; length=N + 1)
        y    = collect(@. 1 - tanh(γ*(1 - ξ))/tanh(γ))
        ends = (NoSymmetry(), EvenSymmetry(1.0))
        D₁   = full(DiffMatrix(y, 5, 1; symmetry=ends))
        D₂   = full(DiffMatrix(y, 5, 2; symmetry=ends))
        return new(1/Reτ, N, y[2:end], D₁[2:end, 2:end], D₂[2:end, 2:end])
    end
end


# ---- the residual: x = [U; ν̃] ----
function residual!(out, x, c::Channel)
    N = c.N
    ν = c.ν
    T = eltype(x)
    U = x[1:N]
    n = x[N + 1:2N]

    # ---- the derivatives, and the eddy viscosity ----
    p   = max.(n, zero(T))
    νt  = p .* fv1.(p ./ ν)
    dU  = c.D₁*U
    d²U = c.D₂*U
    dn  = c.D₁*n
    d²n = c.D₂*n
    dνt = c.D₁*νt

    for i in 1:N
        y = c.y[i]

        # ---- the closures ----
        χ  = p[i]/ν
        f₂ = 1 - χ/(1 + χ*fv1(χ))
        Ω  = abs(dU[i])
        S  = max(Ω + p[i]*f₂/(κ*y)^2, 0.3Ω) + 1e-12
        r  = min(p[i]/(S*(κ*y)^2), one(T)*10)
        g  = r + cw2*(r^6 - r)
        fw = g*((1 + cw3^6)/(g^6 + cw3^6))^(1/6)

        # ---- momentum, then production, destruction and diffusion, with (ν̃ν̃')' = ν̃ ν̃'' + ν̃'² ----
        out[i]     = (ν + νt[i])*d²U[i] + dνt[i]*dU[i] + 1
        out[N + i] = cb1*S*p[i] - cw1*fw*(p[i]/y)^2 + ((ν + p[i])*d²n[i] + (1 + cb2)*dn[i]^2)/σ
    end

    return out
end


# ---- the diffusion d/dy a d/dy as a matrix ----
diffusion(c::Channel, a) = Diagonal(a)*c.D₂ + Diagonal(c.D₁*a)*c.D₁


# ---- the preconditioners: one matrix for each field, the right-hand sides split between them ----
mutable struct Diffusion
    c::Channel # the channel
    A::Any     # LU factorisation of the diffusion of the velocity
    B::Any     # and of that of ν̃

    Diffusion(c::Channel) = new(c, nothing, nothing)
end

function NKRoots.update!(M::Diffusion, x)
    c   = M.c
    p   = max.(x[c.N + 1:2c.N], 0.0)
    νt  = p .* fv1.(p ./ c.ν)
    M.A = lu(diffusion(c, c.ν .+ νt))
    M.B = lu(diffusion(c, (c.ν .+ p) ./ σ))
    return M
end

struct Laplacian
    N::Int # the unknowns of each field
    A::Any # LU factorisation of ν D₂

    Laplacian(c::Channel) = new(c.N, lu(c.ν .* c.D₂))
end

_solve!(out, A, B, v, N) = (out[1:N] .= A \ v[1:N]; out[N + 1:2N] .= B \ v[N + 1:2N]; out)

LinearAlgebra.ldiv!(out, M::Diffusion, v) = _solve!(out, M.A, M.B, v, M.c.N)
LinearAlgebra.ldiv!(out, M::Laplacian, v) = _solve!(out, M.A, M.A, v, M.N)


# ---- a search from a guess ----
function search(Reτ, x₀, precond; N=300, krylov_dim=300, maxiter=100, quiet=false)
    c = Channel(Reτ, N)
    x = copy(x₀)
    P = precond === :none ? nothing : precond === :laplacian ? Laplacian(c) : Diffusion(c)

    trace = Trace()
    out   = nkroot!((o, v) -> residual!(o, v, c), x; jacobian=:ad, precond=P, Δ=1.0, krylov_dim=krylov_dim,
                    krylov_tol=1e-3, tol=1e-8, maxiter=maxiter, verbose=false, callback=trace)

    quiet || @printf "Reτ = %4d, %-9s: %-10s %3d steps, %5d products, ‖F‖ = %.1e\n" Reτ precond out.status out.iter trace.products[end] out.res
    flush(stdout)

    return (; c, x, trace, out)
end


# ---- continuation in the Reynolds number: each solution is the guess of the next ----
function continuation(numbers; N=300, quiet=false)
    c     = Channel(numbers[1], N)
    guess = [@. 20*c.y*(2 - c.y); @. κ*c.y*(1 - c.y) + 3c.ν]
    return map(numbers) do Reτ
        s      = search(Reτ, guess, :diffusion; N=N, quiet=quiet)
        guess .= s.x
        s
    end
end

numbers = (180, 300, 550, 1000, 2000, 5200)
steps   = continuation(numbers)


# ---- the three preconditioners, on the same step of the continuation ----
start = steps[3].x
cases = [search(1000, start, p; maxiter=40) for p in (:none, :laplacian, :diffusion)]


# ---- the error of the total stress at Reτ = 1000, and the order of accuracy ----
function stress(s)
    c  = s.c
    U  = s.x[1:c.N]
    p  = max.(s.x[c.N + 1:2c.N], 0.0)
    νt = p .* fv1.(p ./ c.ν)
    return maximum(abs, (c.ν .+ νt) .* (c.D₁*U) .- (1 .- c.y))
end

# the solution s by linear interpolation on the nodes of N unknowns
function interpolate(s, N)
    c = s.c
    y = Channel(1000, N).y
    f(v) = map(y) do yᵢ
        j = searchsortedfirst(c.y, yᵢ)
        j == 1 && return v[1]*yᵢ/c.y[1]
        j > c.N && return v[c.N]
        t = (yᵢ - c.y[j - 1])/(c.y[j] - c.y[j - 1])
        (1 - t)*v[j - 1] + t*v[j]
    end
    return [f(s.x[1:c.N]); f(s.x[c.N + 1:2c.N])]
end

sizes  = (150, 300, 600)
checks = [search(1000, interpolate(steps[4], N), :diffusion; N=N, quiet=true) for N in sizes]
errors = stress.(checks)
for k in eachindex(sizes)
    rate = k == 1 ? "" : @sprintf("order %.2f", log2(errors[k - 1]/errors[k]))
    @printf "N = %4d: %-10s error of the total stress %.2e, %s\n" sizes[k] checks[k].out.status errors[k] rate
end


# ---- the profiles ----
curves = map(numbers, steps) do Reτ, s
    c = s.c
    (yp=c.y .* Reτ, U=s.x[1:c.N], n=s.x[c.N + 1:2c.N] .* Reτ)
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
