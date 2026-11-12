# The hookstep. From the guess (1.8, 0.7) the Newton step lands above the root and raises the residual.
# A radius Δ bounds the step. As Δ grows the step turns from the direction of steepest descent to the
# Newton step, and the ratio ρ of the actual to the predicted reduction says how far the linear model
# can be trusted.
#
#     julia --project=examples examples/hookstep/hookstep.jl
include(joinpath(@__DIR__, "..", "common.jl"))


# ---- one step from the guess for each radius: Δmax fixes the radius, eta accepts every trial ----
x₀    = [1.8, 0.7]
radii = 10 .^ range(-2, 1; length=100)
steps = map(radii) do Δ
    x = copy(x₀)
    t = Trace()
    nkroot!(two!, x; jacobian=:ad, Δ=Δ, Δmax=Δ, eta=-Inf, maxiter=1, verbose=false, callback=t)
    (x=x, ρ=t.ratio[end])
end


# ---- the figure ----
fig = Figure(size=(780, 380))
ax₁ = Axis(fig[1, 1]; title="the step for each radius", xlabel="x₁", ylabel="x₂", aspect=DataAspect(),
           ygridvisible=false, limits=((0.7, 2.2), (0.3, 1.6)))
landscape!(ax₁, range(0.7, 2.2; length=300), range(0.3, 1.6; length=300))

for Δ in (0.1, 0.25, 0.5)
    arc!(ax₁, Point2f(x₀), Δ, 0, 2π; color=(:black, 0.45), linestyle=:dash, linewidth=1)
end
lines!(ax₁, first.(getfield.(steps, :x)), last.(getfield.(steps, :x)); color=:black)
scatter!(ax₁, [x₀[1]], [x₀[2]]; color=:white, strokecolor=:black, strokewidth=1.5, markersize=11)
scatter!(ax₁, [1.0], [1.0]; color=:black, marker=:star5, markersize=14)
text!(ax₁, 1.86, 0.62; text="guess", color=INK, fontsize=13)
text!(ax₁, 0.97, 1.05; text="root", color=INK, fontsize=13, align=(:right, :bottom))

ax₂ = Axis(fig[1, 2]; title="the ratio ρ", xlabel="radius Δ", ylabel="actual / predicted", xscale=log10,
           xticks=LogTicks(-2:1), limits=((1e-2, 1e1), nothing))
hlines!(ax₂, [0.25, 0.75]; color=(:black, 0.45), linestyle=:dash, linewidth=1)
lines!(ax₂, radii, getfield.(steps, :ρ); color=BLUE)
text!(ax₂, 0.0105, 0.76; text="above 3/4 on the boundary, Δ doubles", color=INK, fontsize=12, align=(:left, :bottom))
text!(ax₂, 0.0105, 0.24; text="below 1/4, Δ shrinks by four", color=INK, fontsize=12, align=(:left, :top))

finish(fig, joinpath(@__DIR__, "hookstep.png"))
