# Equilibria of Kolmogorov flow, after Golden (2026). The flow is two-dimensional and periodic on the
# square [0, 2π)², driven by the force sin(4y) eₓ at Re = 40. An equilibrium is a root of
#
#     N(ω) = u·∇ω - (1/Re) ∇²ω + 4 cos(4y),
#
# with ω the vorticity, on a grid of 128 × 128 points with the two-thirds rule. The paper searches
# travelling waves and equilibria together, with the wave speed as an unknown and a phase condition.
# This example keeps to equilibria. It searches in the subspace of the fields invariant under the
# rotation (x, y) → (-x, -y), where a state cannot drift along x, so that no phase condition is
# needed. The formulation is in notes/kolmogorov/kolmogorov.pdf and the discretisation in problem.jl.
#
# The script checks the residual on the laminar state, runs the same guesses with backtracking and
# with the hookstep under four rules (the classical rule of the radius, the greedy scan, the longest
# scan, and the classical rule with the geodesic acceleration), compares GMRES with and without the
# preconditioner, the Stokes operator, and draws two equilibria.
#
# Settings: Re = 40, 128 × 128 points, guesses of filtered noise with the seeds 1 to 8, the Jacobian
# by finite differences, precond = Stokes, krylov_dim = 200, krylov_tol = 1e-2, tol = 1e-8 in the norm
# of the values on the grid, maxiter = 80, eta = 1e-3, first radius Δ = 20 for the hookstep. The
# cost of a search is counted in calls of G, residuals and Jacobian products together, since a
# product by finite differences is one call of G. The script takes about sixteen minutes, most of
# them in the searches that do not converge.
#
#     julia --project=examples examples/kolmogorov/kolmogorov.jl
include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "problem.jl"))


# ---- the problem ----
K = Kolmogorov(128; Re=40)
G!(out, ω) = residual!(out, ω, K)


# ---- the laminar state solves the equations, and its dissipation and input are Re/32 ----
ωL = laminar(K)
b  = budget(K, ωL)
println("the laminar state")
@printf "   ‖G‖ = %.1e, dissipation %.12f, input %.12f, Re/32 = %.12f\n" norm(G!(similar(ωL), ωL)) b.dissipation b.input K.Re/32


# ---- one search from the guess of a seed ----
function search(seed; keep=false, kwargs...)
    trace = Trace(keep=keep)
    ω     = guess(K, seed)
    time  = @elapsed out = nkroot!(G!, ω; precond=Stokes(K),
                                          Δ=20.0,
                                          krylov_dim=200,
                                          krylov_tol=1e-2,
                                          tol=1e-8,
                                          maxiter=80,
                                          verbose=false,
                                          callback=trace,
                                          kwargs...)
    return (; out, trace, ω, time, seed)
end

# the root of a search: none, the laminar state, or another equilibrium
function kind(s)
    s.out.status === :converged || return "-"
    return norm(s.ω .- ωL) < 1e-6 * norm(ωL) ? "laminar" : "equilibrium"
end


# ---- the same guesses with backtracking and with four rules of the hookstep ----
methods = [(name="globalisation=:backtracking",               short="backtracking",
            kw=(globalisation=:backtracking,)),
           (name="trust=:classical",                          short="hookstep, classical",
            kw=(trust=:classical,)),
           (name="trust=:scan, scan=:greedy",                 short="hookstep, greedy scan",
            kw=(trust=:scan, scan=:greedy)),
           (name="trust=:scan, scan=:longest",                short="hookstep, longest scan",
            kw=(trust=:scan, scan=:longest)),
           (name="trust=:classical, acceleration=:geodesic",  short="hookstep, classical, geodesic",
            kw=(trust=:classical, acceleration=:geodesic))]
seeds   = 1:8

println("\nthe searches")
runs = map(methods) do m
    map(seeds) do seed
        s = search(seed; m.kw...)
        t = s.trace
        e = budget(K, s.ω)
        @printf "   seed %2d %-41s %-9s %3d iterations, %4d residuals, %5d products, %3.0f s, ‖G‖ = %.1e, %-11s D = %.6f, I = %.6f, E = %.6f\n" seed m.name s.out.status s.out.iter t.residuals[end] t.products[end] s.time s.out.res kind(s) e.dissipation e.input e.energy
        flush(stdout)
        s
    end
end


# ---- for each rule: the searches that converge and their cost, where the others end, and the cost of
#      all eight in calls of G, residuals and Jacobian products together ----
println("\nthe searches that converge, of $(length(seeds))")
for (m, list) in zip(methods, runs)
    good  = [s for s in list if s.out.status === :converged]
    bad   = [s for s in list if s.out.status !== :converged]
    marks = join(s.out.status === :converged ? "✓" : "·" for s in list)
    calls = sum(s.trace.function_calls[end] for s in list)
    time  = sum(s.time for s in list)

    @printf "   %-41s %d  %s   all of them: %4d residuals, %6d products, %6d calls of G, %4.0f s\n" m.name length(good) marks sum(s.trace.residuals[end] for s in list) sum(s.trace.products[end] for s in list) calls time
    for s in good
        @printf "      seed %d converges in %2d iterations, %4d residuals, %5d products, %5d calls of G, D = %.6f\n" s.seed s.out.iter s.trace.residuals[end] s.trace.products[end] s.trace.function_calls[end] budget(K, s.ω).dissipation
    end
    isempty(bad) ||
        println("      the others end at ‖G‖ = ", join((@sprintf("%.1f (seed %d, %s)", s.out.res, s.seed, s.out.status) for s in bad), ", "))
end


# ---- the equilibria found, told apart by their dissipation and energy, and their energy balance ----
found = [s for list in runs for s in list if kind(s) == "equilibrium"]
table = unique(s -> round.((budget(K, s.ω).dissipation, budget(K, s.ω).energy); digits=5), found)
println("\nthe equilibria that are not the laminar state")
for s in table
    e     = budget(K, s.ω)
    stray = norm(s.ω .- field(K, project(K, s.ω)))
    rules = [m.short for (m, list) in zip(methods, runs) for t in list
             if kind(t) == "equilibrium" && isapprox(budget(K, t.ω).dissipation, e.dissipation; rtol=1e-6) &&
                isapprox(budget(K, t.ω).energy, e.energy; rtol=1e-6)]
    @printf "   D = %.8f, I = %.8f, |D - I| = %.1e, E = %.8f, part outside the subspace %.1e, found %d times\n" e.dissipation e.input abs(e.dissipation - e.input) e.energy stray length(rules)
end


# ---- GMRES with and without the preconditioner: the first Newton system of the first seed from which
#      the classical rule converges, and the Newton system of the last iteration of that search, each
#      solved in a space of up to 600 vectors. The Krylov space of an iteration is built before the
#      globalisation, so one iteration of any rule gives it ----
classical  = runs[2]
first_seed = first(s.seed for s in classical if s.out.status === :converged)
first_ω    = guess(K, first_seed)
last_ω     = search(first_seed; keep=true).trace.x[end - 1]

function gmres(ω₀, precond)
    trace = Trace()
    nkroot!(G!, copy(ω₀); precond=precond, krylov_dim=600, krylov_tol=1e-8, maxiter=1, verbose=false,
            callback=trace)
    return trace.krylov[2]
end

systems = [(name="the first Newton system", point=first_ω, style=:solid),
           (name="the last Newton system",  point=last_ω,  style=:dash)]
curves  = [(s, with=gmres(s.point, Stokes(K)), without=gmres(s.point, nothing)) for s in systems]
println("\nGMRES to a relative residual of 1e-8, in a space of up to 600 vectors, seed $first_seed")
for c in curves
    @printf "   %-24s with the preconditioner %3d vectors, residual %.1e. Without, %3d vectors, residual %.1e\n" c[1].name length(c.with) c.with[end] length(c.without) c.without[end]
end

# the search of the same seed without the preconditioner, with the classical rule
plain = search(first_seed; precond=nothing)
@printf "   the search of seed %d without the preconditioner: %s after %d iterations, %d products, ‖G‖ = %.1e\n" first_seed plain.out.status plain.out.iter plain.trace.products[end] plain.out.res


# ---- the figure: one panel per rule with the residual of its eight searches against their cost, those
#      that converge dark and the others light; GMRES; the vorticity of two equilibria ----
fig = Figure(size=(1150, 900))

# ---- the cost axis, in thousands of calls of G: ticks at a round step, and a margin beyond the last
#      one so that the labels of neighbouring panels do not meet ----
top   = maximum(s.trace.function_calls[end] for list in runs for s in list)
step  = 5000 * max(1, round(Int, top / 15000))
ticks = 0:step:max(top, step)
right = 1.1 * max(top, step)

for (col, (m, list)) in enumerate(zip(methods, runs))
    good = count(s -> s.out.status === :converged, list)
    ax   = Axis(fig[1, col]; title="$(m.short)\n$good of $(length(list)) converge", xlabel="calls of G, thousands",
                ylabel=col == 1 ? "‖G‖" : "", yscale=log10, yticks=LogTicks(-10:2:4),
                limits=((0, right), (1e-10, 1e4)), xticks=(ticks, string.(ticks .÷ 1000)), titlesize=13)
    col > 1 && hideydecorations!(ax; grid=false, ticks=false)
    for s in list
        ok = s.out.status === :converged
        lines!(ax, s.trace.function_calls, max.(s.trace.res, 1e-10); color=ok ? (BLUE, 1.0) : (INK, 0.35),
               linewidth=ok ? 1.8 : 1.0)
    end
end
colgap!(fig.layout, 14)

# ---- below: GMRES and the vorticity of two equilibria, in three columns of the same width ----
below = fig[2, 1:5] = GridLayout()

ax₂ = Axis(below[1, 1]; title="GMRES, seed $first_seed", xlabel="Arnoldi step", ylabel="relative residual",
           yscale=log10, yticks=LogTicks(-8:2:0), limits=((0, 600), (1e-8, 1e0)))
for c in curves
    lines!(ax₂, eachindex(c.with), c.with; color=ORANGE, linestyle=c[1].style, label="Stokes, $(c[1].name)")
    lines!(ax₂, eachindex(c.without), c.without; color=BLUE, linestyle=c[1].style, label="none, $(c[1].name)")
end
Legend(below[2, 1], ax₂; orientation=:horizontal, tellwidth=false, labelsize=11, nbanks=2)

x = [2π*i/K.N for i in 0:K.N - 1]
for (k, s) in enumerate(table[1:min(2, end)])
    e   = budget(K, s.ω)
    ax  = Axis(below[1, k + 1]; title=@sprintf("the vorticity, D = %.4f, E = %.4f", e.dissipation, e.energy),
               xlabel="x", ylabel="y", aspect=DataAspect(), ygridvisible=false, limits=((0, 2π), (0, 2π)),
               xticks=0:2:6, yticks=0:2:6)
    big = maximum(abs, s.ω)
    hm  = heatmap!(ax, x, x, s.ω; colormap=:balance, colorrange=(-big, big))
    Colorbar(below[2, k + 1], hm; vertical=false, flipaxis=false, width=Relative(0.8), height=10)
end
colsize!(below, 1, Relative(0.38))
rowsize!(fig.layout, 1, Relative(0.45))

finish(fig, joinpath(@__DIR__, "kolmogorov.png"))
