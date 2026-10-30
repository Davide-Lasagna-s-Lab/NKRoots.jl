# Everything a search records. A Trace is the callback of nkroot!, and after the search it holds the
# whole history: the residuals, the Krylov spaces, the trust region, the rejected trials, the counts
# and the wall time. With keep=true it also holds the iterates. Here it follows the Bratu problem
# from a guess far from the root, with a tight tolerance and a preconditioner of the Laplacian alone.
#
#     julia --project=examples examples/tracing.jl
include("common.jl")
include("bratu_problem.jl")


# ---- the search, recorded ----
trace = Trace(keep=true)
u     = [20*sin(π*i*h) for i in 1:n]
out   = nkroot!(bratu!, u; precond=lu(A), Δ=20.0, krylov_tol=1e-6, tol=1e-9, maxiter=60, verbose=false, callback=trace)

display(trace)


# ---- what it holds ----
steps = 1:length(trace) - 1
rows  = steps .+ 1


# ---- the figure ----
fig = Figure(size=(900, 820))

# ---- the residual, against the steps, the products and the time ----
ax = Axis(fig[1, 1]; title="trace.res", xlabel="trace.iter", ylabel="‖F‖", yscale=log10)
scatterlines!(ax, trace.iter, trace.res; color=BLUE)

ax = Axis(fig[1, 2]; title="against trace.products", xlabel="Jacobian products", ylabel="‖F‖", yscale=log10)
scatterlines!(ax, trace.products, trace.res; color=BLUE)

ax = Axis(fig[1, 3]; title="against trace.time", xlabel="seconds", ylabel="‖F‖", yscale=log10)
scatterlines!(ax, trace.time, trace.res; color=BLUE)

# ---- GMRES inside each step, darker with the step ----
ax = Axis(fig[2, 1]; title="trace.krylov", xlabel="Arnoldi step", ylabel="relative residual", yscale=log10,
          yticks=LogTicks(-6:2:0))
for k in rows
    lines!(ax, eachindex(trace.krylov[k]), trace.krylov[k]; color=(k - 1)/maximum(steps), colorrange=(0, 1),
           colormap=[colorant"#b9d6f2", colorant"#123f78"], linewidth=1.5)
end

ax = Axis(fig[2, 2]; title="length.(trace.krylov)", xlabel="Newton step", ylabel="Krylov dimension")
barplot!(ax, steps, length.(trace.krylov[rows]); color=BLUE, gap=0.25)

ax = Axis(fig[2, 3]; title="the cumulative counts", xlabel="Newton step", ylabel="evaluations")
lines!(ax, trace.iter, trace.residuals; color=BLUE, label="residuals")
lines!(ax, trace.iter, trace.products; color=ORANGE, label="products")
lines!(ax, trace.iter, trace.preconditioner; color=AQUA, label="preconditioner")
axislegend(ax; position=(0.95, 0.3), labelsize=12, rowgap=0)

# ---- the trust region ----
ax = Axis(fig[3, 1]; title="trace.radius and trace.steplength", xlabel="Newton step", ylabel="length", yscale=log10)
scatterlines!(ax, steps, trace.radius[rows]; color=BLUE, label="radius")
lines!(ax, steps, trace.steplength[rows]; color=ORANGE, label="step")
axislegend(ax; position=:lb, labelsize=12, rowgap=0)

ax = Axis(fig[3, 2]; title="trace.ratio", xlabel="Newton step", ylabel="ρ")
hlines!(ax, [0.25, 0.75]; color=(:black, 0.45), linestyle=:dash, linewidth=1)
hook = findall(==("hook"), trace.step)
full = findall(==("newton"), trace.step)
scatter!(ax, hook .- 1, trace.ratio[hook]; color=:white, strokecolor=BLUE, strokewidth=1.5, markersize=9, label="hook")
scatter!(ax, full .- 1, trace.ratio[full]; color=BLUE, markersize=9, label="newton")
axislegend(ax; position=:lb, labelsize=12, rowgap=0)

ax = Axis(fig[3, 3]; title="trace.rejected and trace.trials", xlabel="Newton step", ylabel="radius of each trial", yscale=log10,
          yticks=[2.5, 5, 10, 20, 40])
for k in rows, (Δ, ρ) in trace.trials[k]
    scatter!(ax, [k - 1], [Δ]; color=:white, strokecolor=ORANGE, strokewidth=1.5, markersize=9, marker=:xcross)
end
scatterlines!(ax, steps, trace.radius[rows]; color=BLUE, markersize=6)

# ---- the iterates, from trace.x ----
ax = Axis(fig[4, 1:3]; title="trace.x", xlabel="position", ylabel="u", height=170)
for k in eachindex(trace.x)
    lines!(ax, (1:n) .* h, trace.x[k]; color=(k - 1)/(length(trace.x) - 1), colorrange=(0, 1),
           colormap=[colorant"#b9d6f2", colorant"#123f78"], linewidth=1.5)
end

finish(fig, "tracing")
