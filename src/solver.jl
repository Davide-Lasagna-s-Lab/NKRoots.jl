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
# Geodesic acceleration (Transtrum and Sethna 2012, section 3). To second order the residual along a
# step s is F(x + s) ≈ r + J s + F''[s, s]/2. With the hookstep v, the velocity, and its multiplier
# μ, the acceleration a solves the same damped problem with F''[v, v] in place of r,
#
#     (JᵀJ + μ I) a = -Jᵀ F''[v, v],                                                 (4)
#
# and the step is v + a/2. F''[v, v] comes from one more residual, by the forward difference
# (2/h)((F(x + h v) - r)/h - J v) with h = 0.1, and J v = Qₙ₊₁ H y from the Arnoldi relation; (4) is
# solved in the Krylov space, on the coordinates of F''[v, v] on Qₙ₊₁. The step is accepted only if
# ‖a‖ ≤ α ‖v‖ and its ratio, against the reduction predicted for v, exceeds eta. The trust region
# bounds the velocity, and the GMRES step inside it is not accelerated.
#
# The files: krylov.jl, the Arnoldi iteration; preconditioner.jl, the right preconditioner; model.jl,
# the linear model of (1) in the Krylov space; hookstep.jl, the constrained step; globalisation.jl,
# the rules of the step; output.jl, the table; trace.jl, the record of a search; jacobian.jl, the
# Jacobians by finite differences and by dual numbers.

export nkroot!

"""
    nkroot!(F!, J, x; precond=nothing, flexible=false, krylov_dim=50, krylov_tol=1e-3,
            globalisation=:hookstep, trust=:classical, scan=:greedy, acceleration=:none, alpha=0.75,
            Δ=1e-2, Δmax=Inf, eta=1e-3, maxiter=20, tol=1e-10, verbose=true, io=stdout,
            callback=info -> false)
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
  rule; `trust=:scan` chooses the radius among several trials in the same Krylov space: the one of
  smallest residual among thirteen radii, `scan=:greedy`, or the largest radius whose step has a
  ratio above `eta`, found to a factor √2 by expansion and bisection, `scan=:longest`. The longest
  step may lower the residual less now and leave room for longer steps after it.
- `acceleration=:geodesic` adds to every hookstep v the geodesic acceleration of Transtrum and
  Sethna (2012), a/2 with a the damped solution of J a = -F''[v, v], at one more residual per
  trial and no product with `J`; a trial is rejected if ‖a‖ > `alpha` ‖v‖. It needs the hookstep.
- `Δ`, `Δmax`: initial and largest trust-region radius; `Δmax` also caps the step of backtracking.
  `eta`: smallest ratio of the actual to the predicted reduction of `‖r‖²` for a step to be
  accepted, by the hookstep and by backtracking.
- The search stops when `‖r‖ < tol`, after `maxiter` iterations, when no step is accepted, or when
  `callback(info)` returns `true`, called at the start and after every iteration with
  `info = (; iter, x, res, evaluations, krylov, step, radius, steplength, fraction, ratio, rejected,
  trials, globalisation)`:
  the cumulative counts of the calls of the arguments (residuals, Jacobian products, preconditioner
  applications, moves of `J` and of `M`, calls of `gram`, and every call of `F`, those of a Jacobian
  built by NKRoots.jl included), the
  relative residual of the GMRES solution after each Arnoldi step, `"newton"` for the GMRES step,
  `"hook"` for a hookstep on the boundary of the trust region or `"line"` for a GMRES step
  shortened by backtracking, the radius of the accepted step (its length without a trust region),
  its length, the fraction of the GMRES step that it is (its length over that of the GMRES step: 1
  for a `"newton"` step, the λ of backtracking, below 1 for a hookstep unless the geodesic
  acceleration lengthens it; `NaN` at the start), its ratio, the number of rejected trials, the
  radius and ratio of every trial but the accepted one, and the keyword `globalisation`. A
  [`Trace`](@ref) records them.
- With `verbose=true` a table is printed, one row per iteration, with the same columns for every
  globalisation: `‖F‖`, the length `‖δx‖` of the accepted step, the radius `Δ` of the trust region
  after the iteration (blank without a trust region), the fraction `λ`, the ratio `ρ`, the
  dimension of the Krylov space, the relative residual of GMRES, the kind of step and the number of
  rejected trials.

The vectors must broadcast and support `similar`, `dot` and `norm`.
"""
function nkroot!(F!, J, x; precond=nothing,
                           flexible::Bool=false,
                           krylov_dim::Int=50,
                           krylov_tol::Real=1e-3,
                           globalisation::Symbol=:hookstep,
                           trust::Symbol=:classical,
                           scan::Symbol=:greedy,
                           acceleration::Symbol=:none,
                           alpha::Real=0.75,
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
    acceleration in (:none, :geodesic) ||
        throw(ArgumentError("acceleration must be :none or :geodesic"))
    acceleration === :none || globalisation === :hookstep ||
        throw(ArgumentError("acceleration=:geodesic needs globalisation=:hookstep"))

    # ---- the first radius obeys the largest one, as all the others do ----
    Δ = min(Δ, Δmax)

    # ---- the calls of the arguments, counted: residuals, products with J, applications of M⁻¹, moves of J,
    #      moves of M, calls of gram, and the calls of F made by J since the start ----
    counts              = zeros(Int, 6)
    start               = _calls(J)
    residual!(o, v)     = (counts[1] += 1; F!(o, v))
    product!(o, v)      = (counts[2] += 1; mul!(o, J, v))
    precondition!(o, v) = (counts[3] += 1; ldiv!(o, precond, v))
    evaluations()       = (residuals              = counts[1],
                           products               = counts[2],
                           preconditioner         = counts[3],
                           jacobian_updates       = counts[4],
                           preconditioner_updates = counts[5],
                           grams                  = counts[6],
                           function_calls         = counts[1] + _calls(J) - start)

    # ---- the residual at x, in the workspace of the search ----
    ws     = _workspace(x)
    r      = residual!(ws.r, x)
    res    = norm(r)
    status = :maxiter
    iter   = 0

    # ---- the start ----
    region = globalisation === :hookstep    # whether a trust region bounds the step
    verbose && _print_header(io)
    verbose && _print_row(io, 0, res, NaN, region ? Δ : NaN, NaN, NaN, 0, NaN, "", 0)
    stop = callback((; iter, x, res, evaluations=evaluations(), krylov=Float64[], step="", radius=Δ,
                       steplength=NaN, fraction=NaN, ratio=NaN, rejected=0,
                       trials=Tuple{Float64, Float64}[], globalisation))

    while iter < maxiter
        stop && (status = :stopped; break)
        res < tol && (status = :converged; break)
        iter += 1

        # ---- the linear model of the Newton system (1) at x, in a Krylov space ----
        update!(J, x, r)
        counts[4] += 1
        precond === nothing || (update!(precond, x); counts[5] += 1)
        model = _model(r, ws, product!, precondition!; precond=precond,
                                                        flexible=flexible,
                                                        krylov_dim=krylov_dim,
                                                        krylov_tol=krylov_tol)
        precond === nothing || flexible || (counts[3] += _gram_applications(precond, model.dim); counts[6] += 1)

        # ---- the step, by the rule of the globalisation ----
        trial! = (y, μ) -> _trial!(residual!, model, ws, x, r, res, y, μ; accelerate=acceleration === :geodesic,
                                                                        alpha=alpha)
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

        # ---- the length of the step, and the fraction of the GMRES step that it is: 1 for the GMRES
        #      step, the λ of backtracking, the cap of Δmax included, and for a hookstep its length over
        #      that of the GMRES step, with the geodesic acceleration if it has one ----
        steplength = model.len(t.y)
        fraction   = steplength / model.len(model.hook(Inf)[1])

        # ---- output ----
        verbose && _print_row(io, iter, res, steplength, region ? Δ : NaN, fraction, t.ρ, model.dim,
                              model.krylov[end], t.kind, rejected)
        stop = callback((; iter, x, res, evaluations=evaluations(), krylov=model.krylov, step=t.kind,
                           radius=t.used, steplength, fraction, ratio=t.ρ, rejected,
                           trials=t.trials, globalisation))
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
                 jv = similar(x), # J v, for the geodesic acceleration
                 vv = similar(x), # F''[v, v]
                 Q  = _basis(x),  # the Arnoldi vectors
                 W  = _basis(x))  # flexible: the vectors zⱼ = M⁻¹ qⱼ orthonormalised, Zₙ = Wₙ R

# the step of the forward difference of the geodesic acceleration, a tenth of the velocity
const _H = 0.1

# a trial from x, where the residual is r of norm res, with the velocity of Krylov coordinates y and
# multiplier μ: the step is v, or with accelerate=true v + a/2 by (4) for a hookstep on the boundary
# of the trust region, μ > 0, unless ‖a‖ > alpha ‖v‖, which rejects it with no residual of its
# point. The GMRES step inside the region, μ = 0, is taken as it is: near the root it converges
# quadratically, and the difference of (4) along a short v drowns in the noise of F. Its point into
# ws.xn, its residual into ws.rn, the norm of that residual, the ratio of the actual reduction of
# ‖r‖² to the one predicted for v, and the coordinates of the step
function _trial!(residual!, model, ws, x, r, res, y, μ; accelerate::Bool, alpha::Real)
    s = y

    if accelerate && μ > 0
        # ---- F''[v, v] by one more residual, with J v from the Arnoldi relation ----
        model.move!(ws.δx, y)
        ws.xn .= x .+ _H .* ws.δx
        residual!(ws.rn, ws.xn)
        model.jv!(ws.jv, y)
        ws.vv .= (2/_H) .* ((ws.rn .- r) ./ _H .- ws.jv)

        # ---- the acceleration (4) in the Krylov space, on the coordinates of F''[v, v] ----
        c = zeros(model.dim + 1)
        _orthogonalise!(copy(ws.vv), ws.Q, c)
        a = model.damped(μ, .-c)
        model.len(a) <= alpha * model.len(y) ||
            return Inf, -Inf, y
        s = y .+ a ./ 2
    end

    # ---- the residual at the step ----
    model.move!(ws.δx, s)
    ws.xn .= x .+ ws.δx
    resn   = norm(residual!(ws.rn, ws.xn))

    return resn, (res^2 - resn^2) / model.predicted(y), s
end
