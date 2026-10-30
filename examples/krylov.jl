# The Krylov space. The Bratu problem u'' + exp(u) = 0 with u(0) = u(1) = 0 on 200 points. krylov_tol
# sets how well each Newton system is solved. A loose tolerance gives linear convergence with cheap
# steps, a tight one the quadratic convergence of Newton's method. krylov_dim caps the space.
#
#     julia --project=examples examples/krylov.jl
include("common.jl")
include("bratu_problem.jl")


# ---- the searches, with the preconditioner so that the tolerance alone decides ----
function search(; kwargs...)
    trace = Trace()
    u     = zeros(n)
    nkroot!(bratu!, u; precond=lu(A), Δ=1.0, tol=1e-10, maxiter=40, verbose=false, callback=trace, kwargs...)
    return trace
end

tols = (1e-1, 1e-2, 1e-4, 1e-8)
dims = (1, 2, 3, 10)
bytol = [search(krylov_tol=t) for t in tols]
bydim = [search(krylov_dim=d, krylov_tol=1e-8) for d in dims]


# ---- the figure ----
fig = Figure(size=(780, 360))

ax₁ = Axis(fig[1, 1]; title="krylov_tol", xlabel="Jacobian products", ylabel="‖F‖", yscale=log10, yticks=LogTicks(-12:4:4))
for (t, c, l) in zip(bytol, SERIES, string.(tols))
    scatterlines!(ax₁, t.products, t.res; color=c, label=l)
end
axislegend(ax₁; position=:rt)

ax₂ = Axis(fig[1, 2]; title="krylov_dim", xlabel="Newton step", ylabel="‖F‖", yscale=log10, yticks=LogTicks(-12:4:4))
for (t, c, l) in zip(bydim, SERIES, string.(dims))
    scatterlines!(ax₂, 0:length(t.res) - 1, t.res; color=c, label=l)
end
axislegend(ax₂; position=:rt)

finish(fig, "krylov")
