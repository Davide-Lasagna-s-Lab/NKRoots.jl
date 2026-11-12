# The globalisations, on one problem. The Bratu problem u'' + exp(u) = 0 with u(0) = u(1) = 0 on 200
# points, from the guess u = a sin(πx), with the LU factorisation of the second differences as right
# preconditioner and the Jacobian by dual numbers. The same search runs with classical Newton, with
# backtracking, with the hookstep and its three rules of the radius, and with the geodesic
# acceleration. The first part follows the seven searches from a = 10. The second counts, for each
# method, the guesses a = 1, …, 20 from which it converges.
#
# Settings: n = 200, krylov_tol = 1e-6, krylov_dim = 50, tol = 1e-8, eta = 1e-3, maxiter = 100,
# first radius Δ = 1 for the hookstep.
#
#     julia --project=examples examples/globalisation/globalisation.jl
include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "..", "bratu_problem.jl"))


# ---- the methods: name, keywords, colour and line style ----
methods = [(name="globalisation=:none",                   kw=(globalisation=:none,),                                   color=INK,    style=:dot),
           (name="globalisation=:backtracking",           kw=(globalisation=:backtracking,),                           color=INK,    style=:solid),
           (name="trust=:classical",                      kw=(trust=:classical,),                                      color=BLUE,   style=:solid),
           (name="trust=:scan, scan=:greedy",             kw=(trust=:scan, scan=:greedy),                              color=YELLOW, style=:solid),
           (name="trust=:scan, scan=:longest",            kw=(trust=:scan, scan=:longest),                             color=ORANGE, style=:solid),
           (name="trust=:classical, acceleration=:geodesic", kw=(trust=:classical, acceleration=:geodesic),            color=BLUE,   style=:dash),
           (name="scan=:longest, acceleration=:geodesic", kw=(trust=:scan, scan=:longest, acceleration=:geodesic),     color=ORANGE, style=:dash)]


# ---- one search from the guess a sin(πx) ----
function search(a; kwargs...)
    trace = Trace()
    u     = [a*sin(π*i*h) for i in 1:n]
    out   = nkroot!(bratu!, u; jacobian=:ad,
                               precond=lu(A),
                               Δ=1.0,
                               krylov_tol=1e-6,
                               tol=1e-8,
                               maxiter=100,
                               verbose=false,
                               callback=trace,
                               kwargs...)
    return (; out, trace, u)
end


# ---- the seven searches from a = 10 ----
println("from the guess 10 sin(πx)")
runs = map(methods) do m
    s = search(10; m.kw...)
    t = s.trace
    @printf "%-42s %-10s %3d steps, %4d residuals, %4d products, %4d calls of F, ‖F‖ = %.1e\n" m.name s.out.status s.out.iter t.residuals[end] t.products[end] t.function_calls[end] s.out.res
    s
end


# ---- the guesses a = 1, …, 20 from which each method converges ----
println("\nfrom the guesses a sin(πx), a = 1, …, 20")
amplitudes = 1:20
for m in methods
    marks = map(amplitudes) do a
        s = search(a; m.kw...)
        s.out.status === :converged ? "✓" : "·"
    end
    @printf "%-42s %2d of %d  %s\n" m.name count(==("✓"), marks) length(amplitudes) join(marks)
end


# ---- the figure: the residual against the Newton steps and against the calls of F ----
fig = Figure(size=(900, 470))

ax₁ = Axis(fig[1, 1]; title="the residual", xlabel="Newton step", ylabel="‖F‖", yscale=log10,
           yticks=LogTicks(-10:2:8), limits=((0, 45), (1e-10, 1e8)))
ax₂ = Axis(fig[1, 2]; title="the cost", xlabel="calls of F", ylabel="‖F‖", yscale=log10,
           yticks=LogTicks(-10:2:8), limits=((0, 1300), (1e-10, 1e8)))

for (m, s) in zip(methods, runs)
    t = s.trace
    lines!(ax₁, t.iter, t.res; color=m.color, linestyle=m.style, label=m.name)
    lines!(ax₂, t.function_calls, t.res; color=m.color, linestyle=m.style)
end
Legend(fig[2, 1:2], ax₁; orientation=:horizontal, tellwidth=false, labelsize=12, nbanks=3)

finish(fig, joinpath(@__DIR__, "globalisation.png"))
