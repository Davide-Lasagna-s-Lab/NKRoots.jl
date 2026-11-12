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
# The script checks the residual on the laminar state, runs the same guesses with classical Newton,
# with backtracking and with the hookstep, compares GMRES with and without the preconditioner, the
# Stokes operator, and draws two equilibria.
#
# Settings: Re = 40, 128 × 128 points, guesses of filtered noise with the seeds 1 to 8, the Jacobian
# by finite differences, precond = Stokes, krylov_dim = 200, krylov_tol = 1e-2, tol = 1e-8 in the norm
# of the values on the grid, maxiter = 80, eta = 1e-3, first radius Δ = 20 for the hookstep. The
# script takes about seventeen minutes, most of them in the searches that do not converge.
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


# ---- the same guesses with the three globalisations ----
methods = [(name="globalisation=:none",         kw=(globalisation=:none,),         color=INK),
           (name="globalisation=:backtracking", kw=(globalisation=:backtracking,), color=YELLOW),
           (name="globalisation=:hookstep",     kw=(globalisation=:hookstep,),     color=BLUE)]
seeds   = 1:8

println("\nthe searches")
runs = map(methods) do m
    map(seeds) do seed
        s = search(seed; m.kw...)
        e = budget(K, s.ω)
        @printf "   seed %2d %-28s %-9s %3d iterations, %5d products, %3.0f s, ‖G‖ = %.1e, %-11s D = %.6f, I = %.6f, E = %.6f\n" seed m.name s.out.status s.out.iter s.trace.products[end] s.time s.out.res kind(s) e.dissipation e.input e.energy
        flush(stdout)
        s
    end
end

println("\nthe searches that converge, of $(length(seeds))")
for (m, list) in zip(methods, runs)
    marks = [s.out.status === :converged ? "✓" : "·" for s in list]
    @printf "   %-28s %2d  %s\n" m.name count(==("✓"), marks) join(marks)
end

# ---- the equilibria found, told apart by their dissipation and energy, and their energy balance ----
found = [s for list in runs for s in list if kind(s) == "equilibrium"]
table = unique(s -> round.((budget(K, s.ω).dissipation, budget(K, s.ω).energy); digits=5), found)
println("\nthe equilibria that are not the laminar state")
for s in table
    e = budget(K, s.ω)
    stray = norm(s.ω .- field(K, project(K, s.ω)))
    @printf "   seed %2d: D = %.8f, I = %.8f, |D - I| = %.1e, E = %.8f, part outside the subspace %.1e\n" s.seed e.dissipation e.input abs(e.dissipation - e.input) e.energy stray
end


# ---- GMRES with and without the preconditioner: the first Newton system of a guess, and the Newton
#      system of the last iteration of its search, each solved in a space of up to 600 vectors ----
first_guess = guess(K, table[1].seed)
last_point  = search(table[1].seed; keep=true).trace.x[end - 1]

function gmres(ω₀, precond)
    trace = Trace()
    nkroot!(G!, copy(ω₀); precond=precond, globalisation=:none, krylov_dim=600, krylov_tol=1e-8, maxiter=1,
            verbose=false, callback=trace)
    return trace.krylov[2]
end

systems = [(name="the first Newton system", point=first_guess, style=:solid),
           (name="the last Newton system",  point=last_point,  style=:dash)]
curves  = [(s, with=gmres(s.point, Stokes(K)), without=gmres(s.point, nothing)) for s in systems]
println("\nGMRES to a relative residual of 1e-8, in a space of up to 600 vectors")
for c in curves
    @printf "   %-24s with the preconditioner %3d vectors, residual %.1e. Without, %3d vectors, residual %.1e\n" c[1].name length(c.with) c.with[end] length(c.without) c.without[end]
end

# the search of the same seed without the preconditioner
plain = search(table[1].seed; precond=nothing)
@printf "   the search of seed %d without the preconditioner: %s after %d iterations, %d products, ‖G‖ = %.1e\n" table[1].seed plain.out.status plain.out.iter plain.trace.products[end] plain.out.res


# ---- the figure ----
fig = Figure(size=(1000, 860))

# ---- the residual of every search ----
ax₁ = Axis(fig[1, 1]; title="the searches", xlabel="Newton iteration", ylabel="‖G‖", yscale=log10,
           yticks=LogTicks(-10:2:6), limits=((0, 80), (1e-10, 1e6)))
for (m, list) in zip(methods, runs), (i, s) in enumerate(list)
    lines!(ax₁, s.trace.iter, max.(s.trace.res, 1e-10); color=(m.color, 0.7), linewidth=1.2, label=m.name)
end
Legend(fig[2, 1], ax₁; orientation=:horizontal, tellwidth=false, labelsize=12, unique=true, nbanks=2)

# ---- GMRES ----
ax₂ = Axis(fig[1, 2]; title="GMRES with the Stokes preconditioner and with none", xlabel="Arnoldi step", ylabel="relative residual", yscale=log10,
           yticks=LogTicks(-8:2:0), limits=((0, 600), (1e-8, 1e0)))
for c in curves
    lines!(ax₂, eachindex(c.with), c.with; color=ORANGE, linestyle=c[1].style, label="Stokes, $(c[1].name)")
    lines!(ax₂, eachindex(c.without), c.without; color=BLUE, linestyle=c[1].style, label="none, $(c[1].name)")
end
Legend(fig[2, 2], ax₂; orientation=:horizontal, tellwidth=false, labelsize=12, nbanks=2)

# ---- the vorticity of two equilibria ----
x = [2π*i/K.N for i in 0:K.N - 1]
for (col, s) in enumerate(table[1:min(2, end)])
    e  = budget(K, s.ω)
    ax = Axis(fig[3, col]; title=@sprintf("the vorticity, seed %d, D = %.4f, E = %.4f", s.seed, e.dissipation, e.energy),
              xlabel="x", ylabel="y", aspect=DataAspect(), ygridvisible=false, limits=((0, 2π), (0, 2π)))
    top = maximum(abs, s.ω)
    hm  = heatmap!(ax, x, x, s.ω; colormap=:balance, colorrange=(-top, top))
    Colorbar(fig[3, col][1, 2], hm)
end

finish(fig, joinpath(@__DIR__, "kolmogorov.png"))
