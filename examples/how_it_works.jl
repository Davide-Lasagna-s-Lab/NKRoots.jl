# The hookstep with real numbers. The first Newton system of the Bratu problem, from u = 0, its
# Krylov space of 100 vectors, and the singular value decomposition of the small matrix H. The script
# draws the three facts behind the hookstep: the short singular directions carry the long components
# of the GMRES step, the multiplier μ filters them out, and μ is the root of ‖s(μ)‖ = Δ.
#
#     julia --project=examples examples/how_it_works.jl
include("common.jl")
include("bratu_problem.jl")


# ---- the Krylov space, by the Arnoldi iteration of the solver ----
J = Matrix(A) + I
r = bratu!(zeros(n), zeros(n))
β = norm(r)

arnoldi = NKRoots.ArnoldiIteration((out, v) -> mul!(out, J, v), Vector{Float64}[], -r)
for _ in 1:100
    NKRoots.step!(arnoldi)
end
H = arnoldi.H


# ---- the singular value decomposition, and the coordinates of the step ----
U, d, V = svd(H)
e₁      = [1.0; zeros(size(H, 1) - 1)]
p       = β .* (U'*e₁)
s(μ)    = p .* d ./ (d.^2 .+ μ)
ℓ(μ)    = norm(s(μ))


# ---- a radius of one quarter of the GMRES step, and its multiplier by bisection ----
function multiplier(Δ)
    lo = 0.0
    hi = norm(p)*maximum(d)/Δ
    for _ in 1:100
        μ = (lo + hi)/2
        ℓ(μ) > Δ ? (lo = μ) : (hi = μ)
    end
    return hi
end

Δ  = ℓ(0.0)/4
μ★ = multiplier(Δ)

@printf "singular values from %.3g to %.3g, GMRES residual %.2e, step %.3g, radius %.3g, μ = %.3g\n" minimum(d) maximum(d) norm(β*e₁ - H*(V*s(0.0)))/β ℓ(0.0) Δ μ★


# ---- the figure ----
fig = Figure(size=(960, 360))

ax₁ = Axis(fig[1, 1]; title="the GMRES step by singular direction", xlabel="singular value dᵢ", ylabel="|pᵢ/dᵢ|",
           xscale=log10, yscale=log10)
scatter!(ax₁, d, abs.(p ./ d); color=BLUE, markersize=9)

ax₂ = Axis(fig[1, 2]; title="the filter dᵢ²/(dᵢ² + μ)", xlabel="singular value dᵢ", ylabel="what the step keeps",
           xscale=log10)
grid = 10 .^ range(log10(minimum(d)), log10(maximum(d)); length=200)
for (μ, colour, name) in ((μ★/100, AQUA, "μ★/100"), (μ★, ORANGE, "μ★"), (100μ★, YELLOW, "100 μ★"))
    lines!(ax₂, grid, grid.^2 ./ (grid.^2 .+ μ); color=colour, label=name)
end
axislegend(ax₂; position=:rb, labelsize=12, rowgap=0)

ax₃ = Axis(fig[1, 3]; title="the length of the step", xlabel="multiplier μ", ylabel="‖s(μ)‖", xscale=log10, yscale=log10)
μs = 10 .^ range(log10(μ★) - 3, log10(μ★) + 3; length=200)
lines!(ax₃, μs, ℓ.(μs); color=BLUE)
hlines!(ax₃, [Δ]; color=(:black, 0.45), linestyle=:dash, linewidth=1)
scatter!(ax₃, [μ★], [Δ]; color=:white, strokecolor=ORANGE, strokewidth=2, markersize=11)
text!(ax₃, μs[1], Δ; text=" Δ", color=INK, fontsize=13, align=(:left, :bottom))
text!(ax₃, μ★*1.5, Δ*1.1; text="μ★", color=INK, fontsize=13)

finish(fig, "how_it_works")
