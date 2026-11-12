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
# The script checks the residual on the laminar state, runs the same hundred guesses with
# backtracking and with the hookstep under five rules (the classical rule of the radius, the greedy
# scan, the longest scan, and the classical rule and the longest scan with the geodesic
# acceleration), tells apart the equilibria found, compares GMRES with and without the
# preconditioner, the Stokes operator, and draws the equilibrium found most often.
#
# Settings: Re = 40, 128 × 128 points, guesses of filtered noise with the seeds 1 to 100, the
# Jacobian by finite differences, precond = Stokes, krylov_dim = 200, krylov_tol = 1e-2, tol = 1e-8 in
# the norm of the values on the grid, maxiter = 80, eta = 1e-3, first radius Δ = 20 for the hookstep.
# The cost of a search is counted in calls of G, residuals and Jacobian products together, since a
# product by finite differences is one call of G.
#
# The 600 searches run in parallel, one per thread, each on a grid of its own, and a search does not
# depend on the others or on the number of threads. With 8 threads on a laptop of 4 performance and
# 6 efficiency cores the script takes about fifty minutes. Every search is written to searches.csv,
# next to this script.
#
#     julia -t 8 --project=examples examples/kolmogorov/kolmogorov.jl
include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "problem.jl"))

# one thread for each search, and none inside a search: the transforms and the dense algebra run on
# the thread of their search. The number of threads of BLAS changes the rounding of its sums, and with
# it how a search that stalls ends, so it is fixed here
FFTW.set_num_threads(1)
BLAS.set_num_threads(1)


# ---- the problem ----
K = Kolmogorov(128; Re=40)
G!(out, ω) = residual!(out, ω, K)


# ---- the laminar state solves the equations, and its dissipation and input are Re/32 ----
ωL = laminar(K)
b  = budget(K, ωL)
println("the laminar state")
@printf "   ‖G‖ = %.1e, dissipation %.12f, input %.12f, Re/32 = %.12f\n" norm(G!(similar(ωL), ωL)) b.dissipation b.input K.Re/32
flush(stdout)


# ---- one search from the guess of a seed, on the grid K, which holds the work arrays of the search ----
const MAXITER = 80

function search(K, seed; keep=false, kwargs...)
    F!(out, ω) = residual!(out, ω, K)
    trace = Trace(keep=keep)
    ω     = guess(K, seed)
    out   = nkroot!(F!, ω; precond=Stokes(K),
                           Δ=20.0,
                           krylov_dim=200,
                           krylov_tol=1e-2,
                           tol=1e-8,
                           maxiter=MAXITER,
                           verbose=false,
                           callback=trace,
                           kwargs...)
    return (; out, trace, ω, seed)
end


# ---- the rules, and the guesses ----
methods = [(name="globalisation=:backtracking",                 short="backtracking",
            kw=(globalisation=:backtracking,),                   color=INK,    style=:solid),
           (name="trust=:classical",                            short="hookstep, classical",
            kw=(trust=:classical,),                              color=BLUE,   style=:solid),
           (name="trust=:scan, scan=:greedy",                   short="hookstep, greedy scan",
            kw=(trust=:scan, scan=:greedy),                      color=YELLOW, style=:solid),
           (name="trust=:scan, scan=:longest",                  short="hookstep, longest scan",
            kw=(trust=:scan, scan=:longest),                     color=ORANGE, style=:solid),
           (name="trust=:classical, acceleration=:geodesic",    short="hookstep, classical, geodesic",
            kw=(trust=:classical, acceleration=:geodesic),       color=BLUE,   style=:dash),
           (name="scan=:longest, acceleration=:geodesic",       short="hookstep, longest scan, geodesic",
            kw=(trust=:scan, scan=:longest, acceleration=:geodesic), color=ORANGE, style=:dash)]
seeds   = 1:100


# ---- the searches in parallel: each task takes a grid of its own and then the jobs from a queue
#      one at a time, so that a long search does not hold up the others ----
jobs    = [(r, seed) for seed in seeds for r in eachindex(methods)]
results = Vector{Any}(undef, length(jobs))
queue   = Channel{Int}(length(jobs))
foreach(i -> put!(queue, i), eachindex(jobs))
close(queue)

grids = [Kolmogorov(128; Re=40) for _ in 1:Threads.nthreads()]
done  = Threads.Atomic{Int}(0)
clock = @elapsed @sync for grid in grids
    Threads.@spawn for i in queue
        r, seed    = jobs[i]
        results[i] = search(grid, seed; methods[r].kw...)
        k          = Threads.atomic_add!(done, 1) + 1
        k % 60 == 0 && (println(stderr, "   $k of $(length(jobs)) searches"); flush(stderr))
    end
end
runs = [[results[findfirst(==((r, seed)), jobs)] for seed in seeds] for r in eachindex(methods)]
@printf "\n%d searches on %d threads in %.0f minutes\n" length(jobs) Threads.nthreads() clock/60


# ---- the equilibria found: grouped by their dissipation and energy, which agree within a group to
#      far below the tolerance of the search, and checked by the palinstrophy ⟨|∇ω|²⟩, which tells
#      apart two different equilibria that would share the first two ----
function palinstrophy(K, ω)
    a  = project(K, ω)
    ωx = K.iplan * (im .* K.kx .* a)
    ωy = K.iplan * (im .* K.ky .* a)
    return sum(ωx.^2 .+ ωy.^2) / length(ωx)
end

converged = [(r, s) for r in eachindex(methods) for s in runs[r] if s.out.status === :converged]
groups    = Vector{Vector{Any}}()   # each group: the searches that reach one equilibrium
for (r, s) in converged
    e = budget(K, s.ω)
    p = palinstrophy(K, s.ω)
    i = findfirst(g -> isapprox(g[1].D, e.dissipation; rtol=1e-6) && isapprox(g[1].E, e.energy; rtol=1e-6), groups)
    i === nothing && (push!(groups, Any[]); i = length(groups))
    push!(groups[i], (D=e.dissipation, E=e.energy, P=p, I=e.input, rule=r, seed=s.seed, ω=s.ω))
end
sort!(groups; by=g -> -length(g))
laminar_group = findfirst(g -> norm(g[1].ω .- ωL) < 1e-6 * norm(ωL), groups)

# the group of a search: 0 for the laminar state, the rank of its equilibrium otherwise, -1 for none
function group(r, seed)
    i = findfirst(g -> any(x -> x.rule == r && x.seed == seed, g), groups)
    return i === nothing ? -1 : i == laminar_group ? 0 : i
end

println("\nthe equilibria")
@printf "   %d different, from %d searches that converge\n" length(groups) length(converged)
@printf "   the laminar state is among them %d times\n" laminar_group === nothing ? 0 : length(groups[laminar_group])
others = [g for (i, g) in enumerate(groups) if i != laminar_group]
@printf "   the equilibrium found most often, %d times, has D = %.8f and E = %.8f\n" length(others[1]) others[1][1].D others[1][1].E
@printf "   found once: %d equilibria\n" count(g -> length(g) == 1, others)

# ---- the checks of the grouping ----
spread(g, f) = (v = [f(x) for x in g]; (maximum(v) - minimum(v)) / maximum(abs, v))
pairs₁ = [(i, j) for i in eachindex(groups) for j in i + 1:length(groups)]
gap    = minimum(max(abs(groups[i][1].D - groups[j][1].D)/groups[i][1].D,
                     abs(groups[i][1].E - groups[j][1].E)/groups[i][1].E) for (i, j) in pairs₁)
@printf "   within a group: largest relative spread of D %.1e, of E %.1e, of the palinstrophy %.1e\n" maximum(g -> spread(g, x -> x.D), groups) maximum(g -> spread(g, x -> x.E), groups) maximum(g -> spread(g, x -> x.P), groups)
@printf "   between groups: smallest relative distance in D or E %.1e\n" gap
@printf "   largest |D - I| over the searches that converge: %.1e\n" maximum(abs(x.D - x.I) for g in groups for x in g)
rounded = unique([round.((x.D, x.E); digits=5) for g in groups for x in g])
@printf "   the pairs (D, E) rounded to five digits: %d different, for %d groups\n" length(rounded) length(groups)


# ---- for each rule: how many converge, to what, at what cost, and how the others end ----
median(v) = (w = sort(v); n = length(w); isodd(n) ? w[(n + 1) ÷ 2] : (w[n ÷ 2] + w[n ÷ 2 + 1]) / 2)
calls(s)  = s.trace.function_calls[end]

println("\nthe searches, of $(length(seeds)) guesses for each rule")
for (r, m) in enumerate(methods)
    list = runs[r]
    good = [s for s in list if s.out.status === :converged]
    bad  = [s for s in list if s.out.status !== :converged]
    lam  = count(s -> group(r, s.seed) == 0, good)
    eqs  = length(unique(group(r, s.seed) for s in good if group(r, s.seed) > 0))
    @printf "   %-41s %3d converge, %2d to the laminar state, %3d different equilibria\n" m.name length(good) lam eqs
    isempty(good) ||
        @printf "      calls of G of those that converge: median %d, from %d to %d\n" median(calls.(good)) minimum(calls.(good)) maximum(calls.(good))
    @printf "      calls of G of all %d searches: %d, of which %d residuals and %d products\n" length(list) sum(calls.(list)) sum(s.trace.residuals[end] for s in list) sum(s.trace.products[end] for s in list)
    isempty(bad) ||
        @printf "      the others: %d reach %d iterations, %d collapse, median final ‖G‖ %.3g\n" count(s -> s.out.status === :maxiter, bad) MAXITER count(s -> s.out.status === :collapsed, bad) median([s.out.res for s in bad])
end


# ---- the paired counts: the guesses are the same for every rule, so for each pair the number of
#      guesses from which one converges and the other does not, and the two-sided sign test on them ----
ok = [[s.out.status === :converged for s in runs[r]] for r in eachindex(methods)]
function sign_test(b, c)
    n = b + c
    n == 0 && return 1.0
    tail = sum(binomial(big(n), i) for i in 0:min(b, c)) / big(2)^n
    return Float64(min(1, 2tail))
end

println("\nthe guesses from which the rule of the row converges and the rule of the column does not")
@printf "   %-38s" ""
foreach(c -> @printf("%6s", "($c)"), eachindex(methods))
println()
for (r, m) in enumerate(methods)
    @printf "   (%d) %-34s" r m.short
    foreach(c -> @printf("%6s", r == c ? "-" : string(count(ok[r] .& .!ok[c]))), eachindex(methods))
    println()
end
println("\nthe two-sided sign test on those counts, for each pair")
for r in eachindex(methods), c in r + 1:length(methods)
    one   = count(ok[r] .& .!ok[c])
    other = count(ok[c] .& .!ok[r])
    @printf "   (%d) against (%d): %2d against %2d, p = %.2g\n" r c one other sign_test(one, other)
end


# ---- every search, to a file next to the script ----
open(joinpath(@__DIR__, "searches.csv"), "w") do io
    println(io, "rule,seed,status,iterations,residuals,products,calls,final_residual,dissipation,energy,equilibrium")
    for (r, m) in enumerate(methods), s in runs[r]
        e = budget(K, s.ω)
        @printf io "%s,%d,%s,%d,%d,%d,%d,%.6e,%.10f,%.10f,%d\n" replace(m.name, "," => ";") s.seed s.out.status s.out.iter s.trace.residuals[end] s.trace.products[end] calls(s) s.out.res e.dissipation e.energy group(r, s.seed)
    end
end


# ---- GMRES with and without the preconditioner: the first Newton system of the first seed from which
#      the classical rule converges, and the Newton system of the last iteration of that search, each
#      solved in a space of up to 600 vectors. The Krylov space of an iteration is built before the
#      globalisation, so one iteration of any rule gives it ----
classical  = runs[2]
first_seed = first(s.seed for s in classical if s.out.status === :converged)
first_ω    = guess(K, first_seed)
last_ω     = search(K, first_seed; keep=true).trace.x[end - 1]

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
plain = search(K, first_seed; precond=nothing)
@printf "   the search of seed %d without the preconditioner: %s after %d iterations, %d products, ‖G‖ = %.1e\n" first_seed plain.out.status plain.out.iter plain.trace.products[end] plain.out.res


# ---- the figure: one panel per rule with the residual of its searches against their cost, those
#      that converge dark and the others faint; the fraction of the guesses that converge within a
#      cost; GMRES; the vorticity of the equilibrium found most often ----
fig = Figure(size=(1150, 1250))

# ---- the cost axis, in calls of G on a logarithmic scale, from the first residual at one call to
#      the next whole decade above the dearest search, with a margin at both ends so that the tick
#      labels of neighbouring panels do not meet ----
top    = maximum(calls(s) for list in runs for s in list)
decade = ceil(Int, log10(top))
span   = (0.6, 1.6 * 10.0^decade)
costs  = (xscale=log10, xticks=LogTicks(0:decade), limits=(span, (1e-10, 1e4)))

for (r, (m, list)) in enumerate(zip(methods, runs))
    good = count(s -> s.out.status === :converged, list)
    ax   = Axis(fig[1 + (r - 1) ÷ 3, 1 + (r - 1) % 3]; title="$(m.short)\n$good of $(length(list)) converge",
                xlabel="calls of G", ylabel="‖G‖", yscale=log10, yticks=LogTicks(-10:2:4), titlesize=13,
                costs...)
    (r - 1) % 3 == 0 || hideydecorations!(ax; grid=false, ticks=false)
    for s in list
        s.out.status === :converged ||
            lines!(ax, s.trace.function_calls, max.(s.trace.res, 1e-10); color=(INK, 0.15), linewidth=0.6)
    end
    for s in list
        s.out.status === :converged &&
            lines!(ax, s.trace.function_calls, max.(s.trace.res, 1e-10); color=(BLUE, 0.6), linewidth=0.9)
    end
end

# ---- below: the fraction converged within a cost, GMRES and the vorticity, in three columns ----
below = fig[3, 1:3] = GridLayout()

# a window round the data: from the cheapest search that converges to the dearest, with a margin of
# a tenth, and up to the next tenth above the largest fraction. The cost axis keeps its labels at
# whole decades. An end of the window moves to the decade next to it when that decade is within a
# factor 1.3, so that its label shows, and when no decade is in sight the nearer one is taken in.
# Minor ticks at the multiples of the decades mark the scale between the labels
won  = [calls(s) for list in runs for s in list if s.out.status === :converged]
lo   = 0.9 * minimum(won)
hi   = 1.1 * maximum(won)
down = 10.0^floor(log10(minimum(won)))
up   = 10.0^ceil(log10(maximum(won)))
minimum(won) / down < 1.3 && (lo = down / 1.02)
up / maximum(won) < 1.3 && (hi = 1.02 * up)
if ceil(log10(lo)) > floor(log10(hi))
    log10(lo) - floor(log10(lo)) < ceil(log10(hi)) - log10(hi) ? (lo = 10.0^floor(log10(lo)) / 1.02) :
                                                                  (hi = 1.02 * 10.0^ceil(log10(hi)))
end
window = (lo, hi)
reach  = maximum(count(s -> s.out.status === :converged, list) / length(list) for list in runs)
upper  = ceil(10 * reach + 0.2) / 10
minors = [k * 10.0^e for e in floor(Int, log10(lo)):ceil(Int, log10(hi)) for k in 2:9]

ax₁ = Axis(below[1, 1]; title="the fraction of the $(length(seeds)) guesses that converge", xlabel="calls of G",
           ylabel="fraction", xscale=log10, xticks=LogTicks(0:decade), xminorticks=minors,
           xminorticksvisible=true, limits=(window, (0, upper)), yticks=0:0.1:upper)
for (m, list) in zip(methods, runs)
    # a rule that never converges stays on zero, and keeps its place in the legend
    c = sort([calls(s) for s in list if s.out.status === :converged])
    stairs!(ax₁, [window[1]; c; window[2]], [0; (1:length(c)) ./ length(list); length(c)/length(list)];
            step=:post, color=m.color, linestyle=m.style, label=m.short)
end
Legend(below[2, 1], ax₁; orientation=:horizontal, nbanks=3, tellheight=true, tellwidth=false, labelsize=10,
       rowgap=0, colgap=8, framevisible=false)
@printf "\nthe window of the panel of the fraction: calls of G from %.0f to %.0f, fraction from 0 to %.1f\n" window[1] window[2] upper

ax₂ = Axis(below[1, 2]; title="GMRES, seed $first_seed", xlabel="Arnoldi step", ylabel="relative residual",
           yscale=log10, yticks=LogTicks(-8:2:0), limits=((0, 600), (1e-8, 1e0)))
for c in curves
    lines!(ax₂, eachindex(c.with), c.with; color=ORANGE, linestyle=c[1].style, label="Stokes, $(c[1].name)")
    lines!(ax₂, eachindex(c.without), c.without; color=BLUE, linestyle=c[1].style, label="none, $(c[1].name)")
end
axislegend(ax₂; position=:rb, labelsize=10, rowgap=0, framevisible=false)

x   = [2π*i/K.N for i in 0:K.N - 1]
eq  = others[1][1]
ax₃ = Axis(below[1, 3]; title=@sprintf("found most often, D = %.4f, E = %.4f", eq.D, eq.E),
           xlabel="x", ylabel="y", aspect=DataAspect(), ygridvisible=false, limits=((0, 2π), (0, 2π)),
           xticks=0:2:6, yticks=0:2:6)
ωmax = maximum(abs, eq.ω)
hm   = heatmap!(ax₃, x, x, eq.ω; colormap=:balance, colorrange=(-ωmax, ωmax))
Colorbar(below[2, 3], hm; vertical=false, flipaxis=false, width=Relative(0.8), height=10, label="vorticity",
         tellheight=true)

colgap!(fig.layout, 20)
rowsize!(fig.layout, 3, Relative(0.4))

finish(fig, joinpath(@__DIR__, "kolmogorov.png"))
