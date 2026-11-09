# The longest step. Two equations whose residual has a narrow valley along the parabola x₂ = x₁²,
# a thousandth wide, with the root at the origin, at the vertex, where the valley bends the most. The hookstep with trust=:classical
# crawls along the bent valley in short steps, since a long straight step leaves it. A scan that keeps
# the longest step that lowers the residual, scan=:longest, follows the valley in a few long steps.
#
#     julia --project=examples examples/longest.jl
include("common.jl")


# ---- the valley: the parabola, and the same sheared by 10⁻³ ----
function valley!(out, x)
    s      = x[2] - x[1]^2
    out[1] = s
    out[2] = s - 1e-3*x[1]
    return out
end

# half the squared residual, the quantity the search lowers
V(a, b) = (abs2(b - a^2) + abs2(b - a^2 - 1e-3*a))/2


# ---- the classical rule and the two scans, from the same guess ----
function search(; kwargs...)
    trace = Trace(keep=true)
    x     = [-1.0, 1.1]
    out   = nkroot!(valley!, x; jacobian=:ad, Δ=1.0, tol=1e-12, maxiter=500, verbose=false, callback=trace,
                    kwargs...)
    @printf "%-28s %-10s %3d steps, %3d hooksteps, %3d residuals\n" join(("$k=$v" for (k, v) in kwargs), " ") out.status out.iter count(==("hook"), trace.step) trace.residuals[end]
    return trace
end

classical = search(trust=:classical)
greedy    = search(trust=:scan, scan=:greedy)
longest   = search(trust=:scan, scan=:longest)


# ---- the figure ----
fig = Figure(size=(900, 760))

# the routes, with the trust region of every step around the point it starts from, on the window of the
# grid x₁, x₂, with the contours of V down to the level `low`
function routes!(ax, x₁, x₂; low)
    contour!(ax, x₁, x₂, [V(a, b) for a in x₁, b in x₂]; levels=10.0 .^ (low:0.5:0), color=(:gray, 0.35),
             linewidth=0.7)
    lines!(ax, x₁, x₁.^2; color=INK, linewidth=1.2, linestyle=:dash, label="x₂ = x₁², the first equation")
    lines!(ax, x₁, x₁.^2 .+ 1e-3 .* x₁; color=AQUA, linewidth=1.2, linestyle=:dash,
           label="x₂ = x₁² + 10⁻³x₁, the second")
    for (t, c, m, name) in ((classical, BLUE, :circle, "trust=:classical"), (longest, ORANGE, :rect, "scan=:longest"))
        for k in 2:length(t.x)
            arc!(ax, Point2f(t.x[k - 1]), t.radius[k], 0, 2π; color=(c, 0.45), linewidth=0.8)
        end
        scatterlines!(ax, first.(t.x), last.(t.x); color=c, marker=m, markersize=5, linewidth=1.5, label=name)
    end
    return ax
end

# ---- the whole valley, and the window of the closer look ----
x₁  = range(-1.2, 0.5; length=400)
x₂  = range(-0.3, 1.4; length=400)
ax₁ = Axis(fig[1, 1]; title="the routes and their trust regions", xlabel="x₁", ylabel="x₂",
           aspect=DataAspect(), ygridvisible=false, limits=(extrema(x₁), extrema(x₂)))
routes!(ax₁, x₁, x₂; low=-7)
zoom = (x₁=(-0.53, -0.47), x₂=(0.22, 0.28))
poly!(ax₁, Rect2f(zoom.x₁[1], zoom.x₂[1], zoom.x₁[2] - zoom.x₁[1], zoom.x₂[2] - zoom.x₂[1]); color=:transparent,
      strokecolor=:black, strokewidth=1)
scatter!(ax₁, [-1.0], [1.1]; color=:white, strokecolor=:black, strokewidth=1.5, markersize=11)
scatter!(ax₁, [0.0], [0.0]; color=:black, marker=:star5, markersize=14)
text!(ax₁, -0.97, 1.14; text="guess", color=INK, fontsize=13)
text!(ax₁, 0.03, -0.1; text="root", color=INK, fontsize=13)
axislegend(ax₁; position=:rt, labelsize=11, framevisible=true, backgroundcolor=(:white, 0.85),
           framecolor=(:white, 0), patchsize=(18, 8))

# ---- the closer look: the short steps of trust=:classical along the floor ----
ax₄ = Axis(fig[1, 2]; title="a closer look, in the black square", xlabel="x₁", ylabel="x₂", aspect=DataAspect(),
           ygridvisible=false, limits=(zoom.x₁, zoom.x₂))
routes!(ax₄, range(zoom.x₁...; length=300), range(zoom.x₂...; length=300); low=-10)

# ---- the residual and the radius against the Newton step ----
ax₂ = Axis(fig[2, 1]; title="the residual", xlabel="Newton step", ylabel="‖F‖", yscale=log10,
           yticks=LogTicks(-16:4:0))
ax₃ = Axis(fig[2, 2]; title="the radius of the accepted step", xlabel="Newton step", ylabel="Δ",
           yscale=log10, yticks=LogTicks(-8:2:0))
for (t, c, m, name) in ((classical, BLUE, :circle, "trust=:classical"), (greedy, YELLOW, :diamond, "scan=:greedy"),
                        (longest, ORANGE, :rect, "scan=:longest"))
    n = length(t.res) - 1
    scatterlines!(ax₂, 0:n, max.(t.res, 1e-16); color=c, marker=m, markersize=4, linewidth=1.5, label=name)
    scatterlines!(ax₃, 1:n, t.radius[2:end]; color=c, marker=m, markersize=4, linewidth=1.5)
end
Legend(fig[3, 1:2], ax₂; orientation=:horizontal, tellwidth=false, labelsize=12)
rowsize!(fig.layout, 1, Relative(0.58))

finish(fig, "longest")
