# The route of a search. Two equations, the circle x₁² + x₂² = 2 and the curve exp(x₁ - 1) + x₂³ = 2,
# cross at the roots (1, 1) and near (-0.71, 1.22). Each search starts from a guess and walks down the
# landscape of ½‖F‖². Most end at a root. One ends in a valley with no root, a local minimum of ½‖F‖²,
# where the search stops with the status :collapsed or :maxiter.
#
#     julia --project=examples examples/route.jl
include("common.jl")


# ---- one search, its iterates kept by the trace ----
trace = Trace(keep=true)
x     = [1.8, 0.7]
out   = nkroot!(two!, x; Δ=0.1, tol=1e-12, callback=trace)
println(out)
println(x)


# ---- more guesses ----
guesses = ([1.8, 0.7], [-1.5, 0.3], [1.6, -1.2], [0.2, 1.9])
routes  = map(guesses) do guess
    t = Trace(keep=true)
    o = nkroot!(two!, copy(guess); Δ=0.1, tol=1e-12, maxiter=50, verbose=false, callback=t)
    @printf "from (%5.2f, %5.2f): %-10s after %2d steps at (%.4f, %.4f)\n" guess[1] guess[2] o.status o.iter t.x[end][1] t.x[end][2]
    t
end


# ---- the second root, from the search that ends there ----
x₋ = routes[2].x[end]


# ---- the figure ----
fig = Figure(size=(640, 440))
ax  = Axis(fig[1, 1]; xlabel="x₁", ylabel="x₂", aspect=DataAspect(), ygridvisible=false,
           limits=((-2, 2.2), (-1.5, 2.1)))
landscape!(ax, range(-2, 2.2; length=300), range(-1.5, 2.1; length=300))

for (t, guess) in zip(routes, guesses)
    a = first.(t.x)
    b = last.(t.x)
    lines!(ax, a, b; color=:black, linewidth=1.5)
    scatter!(ax, a, b; color=:black, markersize=7)
    scatter!(ax, [guess[1]], [guess[2]]; color=:white, strokecolor=:black, strokewidth=1.5, markersize=11)
end
scatter!(ax, [1.0, x₋[1]], [1.0, x₋[2]]; color=:black, marker=:star5, markersize=15)
text!(ax, -1.9, -1.3; text="x₁² + x₂² = 2", color=INK, fontsize=13)
text!(ax, 1.25, 1.75; text="exp(x₁ - 1) + x₂³ = 2", color=INK, fontsize=13, align=(:center, :bottom))

finish(fig, "route")
