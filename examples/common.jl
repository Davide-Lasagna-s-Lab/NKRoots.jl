# What the examples share: the plots, and the two equations of the first examples.
using CairoMakie
using LinearAlgebra
using Printf
using NKRoots


# ---- the look of the figures ----
const BLUE   = "#2a78d6"
const ORANGE = "#eb6834"
const AQUA   = "#1baf7a"
const YELLOW = "#eda100"
const INK    = "#52514e"
const SERIES = (BLUE, ORANGE, AQUA, YELLOW)

set_theme!(Theme(fontsize=14,
                 Axis=(xgridvisible=false,
                       ygridcolor=(:gray, 0.15),
                       topspinevisible=false,
                       rightspinevisible=false,
                       titlealign=:left,
                       titlefont=:regular,
                       titlesize=14),
                 Legend=(framevisible=false,),
                 Lines=(linewidth=2,),
                 Scatterlines=(linewidth=2, markersize=8)))

figure_path(name) = joinpath(@__DIR__, "figures", "$name.png")

function finish(fig, name)
    mkpath(dirname(figure_path(name)))
    save(figure_path(name), fig; px_per_unit=2)
    return fig
end


# ---- two equations in two unknowns, the circle x₁² + x₂² = 2 and the curve exp(x₁ - 1) + x₂³ = 2 ----
function two!(out, x)
    out[1] = x[1]^2 + x[2]^2 - 2
    out[2] = exp(x[1] - 1) + x[2]^3 - 2
    return out
end

# half the squared residual, the quantity the search lowers
R(a, b) = (abs2(a^2 + b^2 - 2) + abs2(exp(a - 1) + b^3 - 2))/2

# the contours of R and the two curves on which an equation holds
function landscape!(ax, x₁, x₂; levels=10.0 .^ (-2:0.5:3))
    z = [R(a, b) for a in x₁, b in x₂]
    contour!(ax, x₁, x₂, z; levels=levels, color=(:gray, 0.35), linewidth=0.7)
    contour!(ax, x₁, x₂, [a^2 + b^2 - 2 for a in x₁, b in x₂]; levels=[0.0], color=BLUE, linewidth=2)
    contour!(ax, x₁, x₂, [exp(a - 1) + b^3 - 2 for a in x₁, b in x₂]; levels=[0.0], color=ORANGE, linewidth=2)
    return ax
end
