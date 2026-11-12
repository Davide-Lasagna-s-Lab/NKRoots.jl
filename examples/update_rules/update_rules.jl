# The radius of the trust region along a search. With trust=:classical the radius moves after each trial.
# With trust=:scan all the radii from Δ/8 to 8Δ are tried in the same Krylov space, and the best is kept.
# A trial costs one residual and no Jacobian product, so a scan is cheap when residuals are cheap
# against the Krylov space. The options Δ, Δmax and eta set the first radius, the largest radius and
# the smallest ratio for a step to be accepted.
#
#     julia --project=examples examples/update_rules/update_rules.jl
include(joinpath(@__DIR__, "..", "common.jl"))


# ---- a guess far from the roots ----
function search(; guess=[2.5, 2.0], kwargs...)
    trace = Trace(keep=true)
    x     = copy(guess)
    out   = nkroot!(two!, x; jacobian=:ad, tol=1e-12, maxiter=200, verbose=false, callback=trace, kwargs...)
    @printf "%-30s %-10s %3d steps, %3d residuals\n" join(("$k=$v" for (k, v) in kwargs), " ") out.status out.iter trace.residuals[end]
    return trace
end

classical = search(trust=:classical, Δ=1.0)
scan  = search(trust=:scan, Δ=1.0)
first_radii = [search(Δ=Δ) for Δ in (1e-3, 1e-2, 1e-1, 1.0)]
caps        = [search(Δ=1.0, Δmax=Δmax) for Δmax in (0.05, 0.2, 1.0, Inf)]
etas        = [search(guess=[1.8, 0.7], Δ=0.9, eta=η) for η in (1e-3, 0.1, 0.5, 0.9)]


# ---- the figure ----
fig = Figure(size=(900, 680))

# ---- the two rules ----
ax₁ = Axis(fig[1, 1]; title="the route", xlabel="x₁", ylabel="x₂", aspect=DataAspect(), ygridvisible=false,
           limits=((0.5, 2.8), (0.5, 2.3)))
landscape!(ax₁, range(0.5, 2.8; length=300), range(0.5, 2.3; length=300))
for (t, c, m, name) in ((classical, BLUE, :circle, "trust=:classical"), (scan, YELLOW, :rect, "trust=:scan"))
    scatterlines!(ax₁, first.(t.x), last.(t.x); color=c, marker=m, label=name)
end
Legend(fig[2, 1], ax₁; orientation=:horizontal, tellwidth=false, labelsize=12)

ax₂ = Axis(fig[1, 2]; title="the radius of the accepted step", xlabel="Newton step", ylabel="Δ", yscale=log10,
           yticks=LogTicks(-2:1), limits=(nothing, (1e-2, 1e1)))
scatterlines!(ax₂, 1:length(classical.radius) - 1, classical.radius[2:end]; color=BLUE)
scatterlines!(ax₂, 1:length(scan.radius) - 1, scan.radius[2:end]; color=YELLOW, marker=:rect)

ax₃ = Axis(fig[1, 3]; title="the cost", xlabel="residuals", ylabel="‖F‖", yscale=log10, yticks=LogTicks(-16:4:4),
           limits=(nothing, (1e-16, 1e4)))
scatterlines!(ax₃, classical.residuals, max.(classical.res, 1e-16); color=BLUE)
scatterlines!(ax₃, scan.residuals, max.(scan.res, 1e-16); color=YELLOW, marker=:rect)

# ---- the options ----
function panel!(pos, title, traces, labels)
    ax = Axis(fig[3, pos]; title=title, xlabel="Newton step", ylabel="‖F‖", yscale=log10, yticks=LogTicks(-16:4:4),
              limits=(nothing, (1e-16, 1e4)))
    for (t, c, l) in zip(traces, SERIES, labels)
        lines!(ax, 0:length(t.res) - 1, max.(t.res, 1e-16); color=c, label=l)
    end
    Legend(fig[4, pos], ax; orientation=:horizontal, tellwidth=false, labelsize=12, colgap=8, patchsize=(14, 10))
end

panel!(1, "the first radius Δ", first_radii, ["0.001", "0.01", "0.1", "1"])
panel!(2, "the largest radius Δmax", caps, ["0.05", "0.2", "1", "∞"])
panel!(3, "the smallest ratio eta", etas, ["0.001", "0.1", "0.5", "0.9"])

finish(fig, joinpath(@__DIR__, "update_rules.png"))
