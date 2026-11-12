# Three ways to give the Jacobian, on the same two equations: a finite difference, forward (the
# default) or central, dual numbers, and an object of your own. The finite difference with a large relative step ϵ shows what
# an inexact Jacobian costs: the search still converges, because the residual is exact, but it loses
# the fast final phase of Newton's method.
#
#     julia --project=examples examples/jacobians.jl
include("common.jl")


# ---- the Jacobian as an object: mul! applies it, update! moves it ----
struct TwoJacobian
    A::Matrix{Float64} # the matrix at the last point
end

function NKRoots.update!(J::TwoJacobian, x, r)
    J.A[1, 1] = 2x[1]
    J.A[1, 2] = 2x[2]
    J.A[2, 1] = exp(x[1] - 1)
    J.A[2, 2] = 3x[2]^2
    return J
end

LinearAlgebra.mul!(out, J::TwoJacobian, v) = mul!(out, J.A, v)


# ---- the searches ----
function search(; kwargs...)
    trace = Trace()
    x     = [1.8, 0.7]
    nkroot!(two!, x; Δ=0.1, tol=1e-13, maxiter=30, verbose=false, callback=trace, kwargs...)
    return trace
end

fd      = search(jacobian=:fd)
central = search(jacobian=:fd, difference=:central)
ad      = search(jacobian=:ad)
rough   = search(jacobian=:fd, ϵ=1e-2)

mine = Trace()
x    = [1.8, 0.7]
nkroot!(two!, TwoJacobian(zeros(2, 2)), x; Δ=0.1, tol=1e-13, maxiter=30, verbose=false, callback=mine)

for (name, t) in (("jacobian=:fd", fd), ("jacobian=:fd, difference=:central", central), ("jacobian=:ad", ad),
                  ("your own object", mine), ("jacobian=:fd, ϵ=1e-2", rough))
    @printf "%-34s %2d steps, %2d products, %2d calls of F, ‖F‖ = %.1e\n" name t.iter[end] t.products[end] t.function_calls[end] t.res[end]
end


# ---- the figure ----
fig = Figure(size=(640, 360))
ax  = Axis(fig[1, 1]; xlabel="Newton step", ylabel="‖F‖", yscale=log10, yticks=LogTicks(-16:2:2),
           limits=(nothing, (1e-16, 1e2)))

floor(v) = max.(v, 1e-16)
scatterlines!(ax, 0:length(fd.res) - 1, floor(fd.res); color=BLUE, label="jacobian=:fd")
scatterlines!(ax, 0:length(central.res) - 1, floor(central.res); color=INK, label="jacobian=:fd, difference=:central", marker=:cross)
scatterlines!(ax, 0:length(ad.res) - 1, floor(ad.res); color=ORANGE, label="jacobian=:ad", marker=:rect)
scatterlines!(ax, 0:length(mine.res) - 1, floor(mine.res); color=AQUA, label="your own object", marker=:diamond)
scatterlines!(ax, 0:length(rough.res) - 1, floor(rough.res); color=YELLOW, label="jacobian=:fd, ϵ=1e-2", marker=:utriangle)
axislegend(ax; position=:rt)

finish(fig, "jacobians")
