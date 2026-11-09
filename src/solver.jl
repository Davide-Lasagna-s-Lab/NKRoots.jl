# Newton–Krylov for F(x) = 0, with the hookstep of Viswanath (2009, section 5,
# https://arxiv.org/abs/0809.1498) as the NewtonHookstep of ReSolver.jl, for any vector type.
#
# Newton's method solves F(x) = 0 by repeatedly solving the linear system
#
#     J δx = -r,     r = F(x),                                                       (1)
#
# with J the Jacobian of F at x, here in a Krylov space (model.jl). Far from a root the full Newton
# step is unreliable, and the step is globalised (globalisation.jl). The hookstep restricts it to a
# trust region ‖δx‖ ≤ Δ, inside which the linear model r + J δx is trusted,
#
#     min ‖r + J δx‖   subject to   ‖δx‖ ≤ Δ,                                       (2)
#
# solved in the Krylov space (hookstep.jl). Backtracking shortens the GMRES step instead, and
# classical Newton takes it as it is. A trial step is judged by the ratio ρ of the actual to the
# predicted reduction of ‖r‖².
#
# The files: krylov.jl, the Arnoldi iteration; preconditioner.jl, the right preconditioner; model.jl,
# the linear model of (1) in the Krylov space; hookstep.jl, the constrained step; globalisation.jl,
# the rules of the step; output.jl, the table; trace.jl, the record of a search; jacobian.jl, the
# Jacobians by finite differences and by dual numbers.

export nkroot!

"""
    nkroot!(F!, J, x; precond=nothing, flexible=false, krylov_dim=50, krylov_tol=1e-3,
            globalisation=:hookstep, trust=:classical, scan=:greedy, Δ=1e-2, Δmax=Inf, eta=1e-3,
            maxiter=20, tol=1e-10, verbose=true, io=stdout, callback=info -> false)
            -> (; status, iter, res)

Solve `F(x) = 0` by Newton–Krylov from the guess `x`, overwritten with the root.

- `F!(out, x)` writes the residual into `out` and returns it.
- `J` is the Jacobian: `mul!(out, J, v)` applies it, and `update!(J, x, r)` linearises it at `x`,
  with `r = F(x)`, at the start of every Newton iteration (see [`jacobianFD`](@ref)).
- `precond`, a right preconditioner `M`: `ldiv!(out, M, v)` applies `M⁻¹`, `update!(M, x)`
  linearises it at `x` with the Jacobian, and `gram(M, Q)`, the Gram matrix of the vectors
  `M⁻¹ Q[j]` or its upper triangular root, gives the length of a step (the default applies `M⁻¹`
  to each). The trust region bounds the step in `x`, with or without preconditioner.
- `flexible=true` keeps the vectors `M⁻¹ qⱼ` of the Arnoldi iteration, orthonormalised: the step
  and its length come from them, at the cost of one more vector per Krylov dimension and with no
  `gram`. Use it with a preconditioner that solves a few directions exactly, which stretches some
  `M⁻¹ qⱼ` far beyond the others: a Gram matrix loses the short ones.
- `krylov_dim`, `krylov_tol`: largest dimension of the Krylov space, and relative residual of the
  GMRES solution of the Newton system at which it stops growing.
- `globalisation=:hookstep` restricts the step to a trust region; `:backtracking` halves the GMRES
  step until its ratio exceeds `eta`; `:none` takes the GMRES step as it is, classical Newton.
- `trust=:classical` moves the radius of the hookstep after each trial by its ratio, the classical
  rule; `trust=:scan` tries several radii in the same Krylov space and keeps, among the trials with
  a ratio above `eta`, the one of smallest residual, `scan=:greedy`, or the longest step,
  `scan=:longest`, which may lower the residual less now and leave room for longer steps after it.
- `Δ`, `Δmax`: initial and largest trust-region radius; `Δmax` also caps the step of backtracking.
  `eta`: smallest ratio of the actual to the predicted reduction of `‖r‖²` for a step to be
  accepted, by the hookstep and by backtracking.
- The search stops when `‖r‖ < tol`, after `maxiter` iterations, when no step is accepted, or when
  `callback(info)` returns `true`, called at the start and after every iteration with
  `info = (; iter, x, res, evaluations, krylov, step, radius, steplength, ratio, rejected, trials)`:
  the cumulative counts of residuals, Jacobian products and preconditioner applications, the
  relative residual of the GMRES solution after each Arnoldi step, `"newton"` for the GMRES step,
  `"hook"` for a hookstep on the boundary of the trust region or `"line"` for a GMRES step
  shortened by backtracking, the radius of the accepted step (its length without a trust region),
  its length, its ratio, the number of rejected trials and the radius and ratio of every trial but
  the accepted one. A [`Trace`](@ref) records them.

The vectors must broadcast and support `similar`, `dot` and `norm`.
"""
function nkroot!(F!, J, x; precond=nothing,
                           flexible::Bool=false,
                           krylov_dim::Int=50,
                           krylov_tol::Real=1e-3,
                           globalisation::Symbol=:hookstep,
                           trust::Symbol=:classical,
                           scan::Symbol=:greedy,
                           Δ::Real=1e-2,
                           Δmax::Real=Inf,
                           eta::Real=1e-3,
                           maxiter::Int=20,
                           tol::Real=1e-10,
                           verbose::Bool=true,
                           io::IO=stdout,
                           callback=info -> false)

    # ---- input checks ----
    globalisation in (:hookstep, :backtracking, :none) ||
        throw(ArgumentError("globalisation must be :hookstep, :backtracking or :none"))
    trust in (:classical, :scan) ||
        throw(ArgumentError("trust must be :classical or :scan"))
    scan in (:greedy, :longest) ||
        throw(ArgumentError("scan must be :greedy or :longest"))

    # ---- the first radius obeys the largest one, as all the others do ----
    Δ = min(Δ, Δmax)

    # ---- evaluations, counted ----
    counts              = [0, 0, 0] # residuals, Jacobian products, preconditioner applications
    residual!(o, v)     = (counts[1] += 1; F!(o, v))
    product!(o, v)      = (counts[2] += 1; mul!(o, J, v))
    precondition!(o, v) = (counts[3] += 1; ldiv!(o, precond, v))
    evaluations()       = (residuals=counts[1], products=counts[2], preconditioner=counts[3])

    # ---- the residual at x, in the workspace of the search ----
    ws     = _workspace(x)
    r      = residual!(ws.r, x)
    res    = norm(r)
    status = :maxiter
    iter   = 0

    # ---- the start ----
    verbose && _print_header(io, globalisation)
    verbose && _print_row(io, 0, res, Δ, NaN, 0, NaN, "", 0)
    stop = callback((; iter, x, res, evaluations=evaluations(), krylov=Float64[], step="", radius=Δ,
                       steplength=NaN, ratio=NaN, rejected=0, trials=Tuple{Float64, Float64}[]))

    while iter < maxiter
        stop && (status = :stopped; break)
        res < tol && (status = :converged; break)
        iter += 1

        # ---- the linear model of the Newton system (1) at x, in a Krylov space ----
        update!(J, x, r)
        precond === nothing || update!(precond, x)
        model = _model(r, ws, product!, precondition!; precond=precond,
                                                        flexible=flexible,
                                                        krylov_dim=krylov_dim,
                                                        krylov_tol=krylov_tol)
        precond === nothing || flexible || (counts[3] += _gram_applications(precond, model.dim))

        # ---- the step, by the rule of the globalisation ----
        trial! = y -> _trial!(residual!, model, ws, x, res, y)
        t      = _step(trial!, model, ws, Δ; globalisation=globalisation,
                                             trust=trust,
                                             scan=scan,
                                             Δmax=Δmax,
                                             eta=eta,
                                             Δmin=eps() * (1 + norm(x)))
        Δ      = t.Δ
        t.accepted || (status = :collapsed; break)

        # ---- the new point ----
        x       .= t.x
        r       .= t.r
        res      = t.res
        rejected = count(q -> !(q[2] > eta), t.trials)

        # ---- output ----
        verbose && _print_row(io, iter, res, globalisation === :hookstep ? Δ : t.λ, t.ρ, model.dim,
                              model.krylov[end], t.kind, rejected)
        stop = callback((; iter, x, res, evaluations=evaluations(), krylov=model.krylov, step=t.kind,
                           radius=t.used, steplength=model.len(t.y), ratio=t.ρ, rejected,
                           trials=t.trials))
    end

    res < tol && (status = :converged)
    verbose && _print_footer(io, status, iter, res, evaluations(), globalisation)

    return (; status, iter, res)
end

"""
    nkroot!(F!, x; jacobian=:fd, difference=:forward, ϵ=nothing, kwargs...)

As `nkroot!(F!, J, x; kwargs...)` with the Jacobian by finite differences of `F!` (`jacobian=:fd`),
forward or central as `difference` says, or by dual numbers (`jacobian=:ad`). The finite difference
takes its relative step from the machine precision, unless `ϵ` sets it (see [`jacobianFD`](@ref)).
"""
function nkroot!(F!, x; jacobian::Symbol=:fd,
                        difference::Symbol=:forward,
                        ϵ::Union{Real, Nothing}=nothing,
                        kwargs...)
    # ---- input checks ----
    jacobian in (:fd, :ad) ||
        throw(ArgumentError("jacobian must be :fd or :ad"))

    J = jacobian === :ad ? jacobianAD(F!, x) : jacobianFD(F!, x; difference=difference, ϵ=ϵ)
    return nkroot!(F!, J, x; kwargs...)
end

# the vectors of a search and its Krylov bases, kept from one iteration to the next
_workspace(x) = (r  = similar(x), # residual at x
                 rn = similar(x), # residual at the trial point
                 b  = similar(x), # right-hand side of (1)
                 z  = similar(x), # preconditioned step Qₙ y
                 w  = similar(x), # M⁻¹ v inside the Arnoldi operator
                 zw = similar(x), # M⁻¹ v of flexible GMRES
                 δx = similar(x), # step
                 xn = similar(x), # trial point x + δx
                 xb = similar(x), # best trial point of a scan
                 rb = similar(x), # and its residual
                 Q  = _basis(x),  # the Arnoldi vectors
                 W  = _basis(x))  # flexible: the vectors zⱼ = M⁻¹ qⱼ orthonormalised, Zₙ = Wₙ R

# a trial step of Krylov coordinates y from x, where the residual norm is res: its point into
# ws.xn, its residual into ws.rn, the norm of that residual and the ratio of the actual to the
# predicted reduction of ‖r‖²
function _trial!(residual!, model, ws, x, res, y)
    model.move!(ws.δx, y)
    ws.xn .= x .+ ws.δx
    resn   = norm(residual!(ws.rn, ws.xn))

    return resn, (res^2 - resn^2) / model.predicted(y)
end
