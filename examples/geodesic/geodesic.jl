# The geodesic acceleration. The valley of examples/longest/longest.jl, along the parabola x₂ = x₁², a
# thousandth wide, with the root at the vertex. The hookstep with the classical rule crawls along it.
# With acceleration=:geodesic every hookstep on the boundary of the trust region, the velocity, gets a
# second-order correction that bends it with the valley, and the search reaches the root in a few
# steps. The trust region bounds the velocity, so that a corrected step may leave it.
#
#     julia --project=examples examples/geodesic/geodesic.jl
include(joinpath(@__DIR__, "..", "common.jl"))


# ---- the valley: the parabola, and the same sheared by 10⁻³ ----
function valley!(out, x)
    s      = x[2] - x[1]^2
    out[1] = s
    out[2] = s - 1e-3*x[1]
    return out
end

# half the squared residual, the quantity the search lowers
V(a, b) = (abs2(b - a^2) + abs2(b - a^2 - 1e-3*a))/2


# ---- the classical rule and the scan of the longest step, without and with the acceleration ----
function search(; kwargs...)
    trace = Trace(keep=true)
    x     = [-1.0, 1.1]
    out   = nkroot!(valley!, x; jacobian=:ad, Δ=1.0, tol=1e-12, maxiter=500, verbose=false, callback=trace,
                    kwargs...)
    @printf "%-50s %-10s %3d steps, %3d residuals, %3d products\n" join(("$k=$v" for (k, v) in kwargs), " ") out.status out.iter trace.residuals[end] trace.products[end]
    return trace
end

runs = [(search(trust=:classical), BLUE, "trust=:classical"),
        (search(trust=:classical, acceleration=:geodesic), YELLOW, "trust=:classical, acceleration=:geodesic"),
        (search(trust=:scan, scan=:longest), ORANGE, "scan=:longest"),
        (search(trust=:scan, scan=:longest, acceleration=:geodesic), AQUA, "scan=:longest, acceleration=:geodesic")]
over = count(k -> runs[2][1].steplength[k] > runs[2][1].radius[k], 2:length(runs[2][1].res))
println("accelerated steps longer than their trust region: $over")


# ---- the figure ----
fig = Figure(size=(1050, 420))

# ---- (a) the routes of the classical rule, without and with the acceleration, and their trust regions ----
x₁ = range(-1.2, 0.5; length=400)
x₂ = range(-0.3, 1.4; length=400)
ax = Axis(fig[1, 1]; title="the routes and their trust regions", xlabel="x₁", ylabel="x₂", aspect=DataAspect(),
          ygridvisible=false, limits=(extrema(x₁), extrema(x₂)))
contour!(ax, x₁, x₂, [V(a, b) for a in x₁, b in x₂]; levels=10.0 .^ (-7:0.5:0), color=(:gray, 0.35), linewidth=0.7)
lines!(ax, x₁, x₁.^2; color=INK, linewidth=1.2, linestyle=:dash)
lines!(ax, x₁, x₁.^2 .+ 1e-3 .* x₁; color=INK, linewidth=1.2, linestyle=:dot)
for (t, c, name) in runs[1:2]
    for k in 2:length(t.x)
        arc!(ax, Point2f(t.x[k - 1]), t.radius[k], 0, 2π; color=(c, 0.5), linewidth=0.8)
    end
    scatterlines!(ax, first.(t.x), last.(t.x); color=c, markersize=5, linewidth=1.5)
end
scatter!(ax, [-1.0], [1.1]; color=:white, strokecolor=:black, strokewidth=1.5, markersize=11)
scatter!(ax, [0.0], [0.0]; color=:black, marker=:star5, markersize=14)
text!(ax, -0.97, 1.14; text="guess", color=INK, fontsize=13)
text!(ax, 0.03, -0.1; text="root", color=INK, fontsize=13)

# ---- (b) the residual against the Newton step, and (c) against the evaluations, residuals and products ----
ticks = ([1, 10, 100, 1000], ["1", "10", "100", "1000"])
bx    = Axis(fig[1, 2]; title="the residual", xlabel="Newton steps", ylabel="‖F‖", xscale=log10, yscale=log10,
             xticks=ticks, yticks=LogTicks(-16:4:0))
cx    = Axis(fig[1, 3]; title="the cost", xlabel="residuals and Jacobian products", ylabel="‖F‖", xscale=log10,
             yscale=log10, xticks=ticks, yticks=LogTicks(-16:4:0))
for (t, c, name) in runs
    n = length(t.res) - 1
    scatterlines!(bx, 1:n, max.(t.res[2:end], 1e-16); color=c, markersize=4, linewidth=1.5, label=name)
    scatterlines!(cx, t.residuals .+ t.products, max.(t.res, 1e-16); color=c, markersize=4, linewidth=1.5)
end
Legend(fig[2, 1:3], bx; orientation=:horizontal, tellwidth=false, labelsize=12, nbanks=2)
colsize!(fig.layout, 1, Relative(0.36))

finish(fig, joinpath(@__DIR__, "geodesic.png"))
