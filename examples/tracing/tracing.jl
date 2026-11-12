# Everything a search records. A Trace is the callback of nkroot!, and after the search it holds the
# whole history: the residuals, the Krylov spaces, the trust region, the rejected trials, the counts
# and the wall time. With keep=true it also holds the iterates. Here it follows the Bratu problem
# from a guess far from the root, with a tight tolerance and a preconditioner of the Laplacian alone.
# The script then reads the trace of a search with backtracking, and stops a search with a callback.
#
# Settings: n = 200, guess 20 sin(πx), finite-difference Jacobian, precond = lu(A), Δ = 20,
# krylov_tol = 1e-6, krylov_dim = 50, tol = 1e-9, maxiter = 60.
#
#     julia --project=examples examples/tracing/tracing.jl
include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "..", "bratu_problem.jl"))


# ---- the search, recorded; a first run compiles it, so that the times are those of the search ----
function search(trace)
    u = [20*sin(π*i*h) for i in 1:n]
    return nkroot!(bratu!, u; precond=lu(A),
                              Δ=20.0,
                              krylov_tol=1e-6,
                              tol=1e-9,
                              maxiter=60,
                              verbose=false,
                              callback=trace)
end

search(Trace())
trace = Trace(keep=true)
out   = search(trace)

display(trace)
println()


# ---- one record: the iteration with the first rejected trial ----
k = findfirst(>(0), trace.rejected)
println("record $k, iteration $(trace.iter[k])")
println("  step       = ", trace.step[k])
println("  radius     = ", trace.radius[k])
println("  steplength = ", trace.steplength[k])
println("  ratio      = ", trace.ratio[k])
println("  trials     = ", trace.trials[k])
println("  rejected   = ", trace.rejected[k])
println("  krylov     = ", length(trace.krylov[k]), " Arnoldi steps, down to ", trace.krylov[k][end])
println("counts at the end")
for name in (:residuals, :products, :preconditioner, :jacobian_updates, :preconditioner_updates, :grams, :function_calls)
    println("  ", rpad(name, 22), " = ", getfield(trace, name)[end])
end


# ---- backtracking: the fraction of the GMRES step taken, and the trials refused before it ----
line = Trace()
v    = fill(10.0, 3)
nkroot!((out, x) -> (out .= atan.(x); out), v; jacobian=:ad,
                                               globalisation=:backtracking,
                                               maxiter=50,
                                               verbose=false,
                                               callback=line)
println("backtracking on arctan from 10")
println("  step     = ", line.step[2:end])
println("  fraction = ", line.fraction[2:end])
println("  refused  = ", length.(line.trials[2:end]))


# ---- a callback of your own: record, and stop below a residual or after some iterations ----
record = Trace()
stop   = info -> (record(info); info.res < 1e-3 || info.iter >= 15)

u    = zeros(n)
file = tempname()
out  = open(file, "w") do io
    nkroot!(bratu!, u; precond=lu(A), Δ=1.0, tol=1e-10, maxiter=20, io=io, callback=stop)
end
println(out)
println(last(readlines(file)))


# ---- what it holds ----
steps = 1:length(trace) - 1
rows  = steps .+ 1


# ---- the figure ----
fig = Figure(size=(900, 820))

# ---- the residual, against the steps, the products and the time ----
decades = (yscale=log10, yticks=LogTicks(-10:4:10), limits=(nothing, (1e-10, 1e10)))

ax = Axis(fig[1, 1]; title="trace.res", xlabel="trace.iter", ylabel="‖F‖", decades...)
scatterlines!(ax, trace.iter, trace.res; color=BLUE)

ax = Axis(fig[1, 2]; title="against trace.products", xlabel="Jacobian products", ylabel="‖F‖", decades...)
scatterlines!(ax, trace.products, trace.res; color=BLUE)

ax = Axis(fig[1, 3]; title="against trace.time", xlabel="seconds", ylabel="‖F‖", decades...)
scatterlines!(ax, trace.time, trace.res; color=BLUE)

# ---- GMRES inside each step, darker with the step ----
ax = Axis(fig[2, 1]; title="trace.krylov", xlabel="Arnoldi step", ylabel="relative residual", yscale=log10,
          yticks=LogTicks(-8:2:0), limits=(nothing, (1e-8, 1e0)))
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
ax = Axis(fig[3, 1]; title="trace.radius and trace.steplength", xlabel="Newton step", ylabel="length", yscale=log10,
          yticks=LogTicks(-8:2:2), limits=(nothing, (1e-8, 1e2)))
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

ax = Axis(fig[3, 3]; title="trace.rejected and trace.trials", xlabel="Newton step", ylabel="radius of each trial",
          yticks=0:10:40, limits=(nothing, (0, 45)))
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

finish(fig, joinpath(@__DIR__, "tracing.png"))
