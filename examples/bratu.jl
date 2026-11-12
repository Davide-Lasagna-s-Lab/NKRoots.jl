# What comes out of the Bratu problem. The equation u'' + exp(u) = 0 with u(0) = u(1) = 0 has two
# solutions, and the guess decides which one a search finds. Both are known in closed form,
#
#     u(x) = -2 log( cosh((x - 1/2) θ/2) / cosh(θ/4) ),     θ = √2 cosh(θ/4),
#
# which checks the searches.
#
#     julia --project=examples examples/bratu.jl
include("common.jl")
include("bratu_problem.jl")


# ---- the two values of θ, by the same solver ----
closed!(out, θ) = (out[1] = θ[1] - √2*cosh(θ[1]/4); out)

θ₁ = [1.5]
θ₂ = [10.0]
nkroot!(closed!, θ₁; Δ=0.5, tol=1e-13, maxiter=50, verbose=false)
nkroot!(closed!, θ₂; Δ=0.5, tol=1e-13, maxiter=50, verbose=false)

exact(θ) = [-2log(cosh((i*h - 1/2)*θ/2)/cosh(θ/4)) for i in 1:n]


# ---- the two searches: from zero, and from a tall guess ----
xs = (1:n) .* h

low       = zeros(n)
low_trace = Trace()
nkroot!(bratu!, low; precond=lu(A), Δ=1.0, krylov_tol=1e-6, tol=1e-8, maxiter=60, verbose=false, callback=low_trace)

high       = [6*sin(π*x) for x in xs]
guess      = copy(high)
high_trace = Trace()
nkroot!(bratu!, high; precond=lu(A), Δ=1.0, krylov_tol=1e-6, tol=1e-8, maxiter=60, verbose=false, callback=high_trace)

@printf "lower solution: max u = %.6f, %d steps, error against the closed form %.1e\n" maximum(low) length(low_trace) - 1 norm(low - exact(θ₁[1]), Inf)
@printf "upper solution: max u = %.6f, %d steps, error against the closed form %.1e\n" maximum(high) length(high_trace) - 1 norm(high - exact(θ₂[1]), Inf)


# ---- the figure ----
fig = Figure(size=(780, 360))

ax₁ = Axis(fig[1, 1]; title="the two solutions", xlabel="x", ylabel="u")
lines!(ax₁, xs, guess; color=:gray55, linestyle=:dash, linewidth=1.5, label="tall guess")
lines!(ax₁, xs, high; color=ORANGE, label="upper solution")
lines!(ax₁, xs, low; color=BLUE, label="lower solution")
axislegend(ax₁; position=:rt, labelsize=12, rowgap=0)

ax₂ = Axis(fig[1, 2]; title="the two searches", xlabel="Newton step", ylabel="‖F‖", yscale=log10,
           yticks=LogTicks(-10:2:6), limits=(nothing, (1e-10, 1e6)))
scatterlines!(ax₂, 0:length(low_trace) - 1, low_trace.res; color=BLUE, label="from zero")
scatterlines!(ax₂, 0:length(high_trace) - 1, high_trace.res; color=ORANGE, marker=:rect, label="from the tall guess")
axislegend(ax₂; position=:lb, labelsize=12, rowgap=0)

finish(fig, "bratu")
