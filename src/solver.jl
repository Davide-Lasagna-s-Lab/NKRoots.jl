# Newton–Krylov hookstep for F(x) = 0 (Viswanath 2009, section 5, https://arxiv.org/abs/0809.1498), as the
# NewtonHookstep of ReSolver.jl, for any vector type.
#
# Newton's method solves F(x) = 0 by repeatedly solving the linear system
#
#     J δx = -r,     r = F(x),                                                       (1)
#
# with J the Jacobian of F at x. Far from a root the full Newton step is unreliable, so the step is
# restricted to a trust region ‖δx‖ ≤ Δ, inside which the linear model r + J δx is trusted:
#
#     min ‖r + J δx‖   subject to   ‖δx‖ ≤ Δ.                                       (2)
#
# Krylov space. Only the action of J is used. The Arnoldi iteration started from b = -r builds an
# orthonormal basis Qₙ of the Krylov space and the (n+1) × n Hessenberg matrix H with
# J Qₙ = Qₙ₊₁ H. For δx = Qₙ y, ‖r + J δx‖ = ‖g - H y‖ with g = (‖b‖, 0, …, 0) and ‖δx‖ = ‖y‖, so
# that (2) in the Krylov space is the small problem
#
#     min ‖g - H y‖   subject to   ‖y‖ ≤ Δ,                                          (3)
#
# solved from the SVD of H (hookstep.jl). The Krylov space stops growing when the residual of the
# unconstrained least-squares (GMRES) solution falls below `krylov_tol` ‖b‖, monitored by Givens
# rotations of H at no cost, or at dimension `krylov_dim`. The criterion is on the unconstrained
# problem on purpose: on the boundary of the trust region the residual of the hookstep stalls at a
# level set by Δ, which no larger space can lower.
#
# Right preconditioning. With a preconditioner M the Arnoldi iteration runs on J M⁻¹ and the step
# is δx = M⁻¹ Qₙ y: the residual of the model is unchanged, and so is the meaning of `krylov_tol`.
# The trust region still bounds the step δx, not its Krylov coordinates: ‖δx‖ = ‖R y‖ with
# Rᵀ R = G the Gram matrix of the vectors zⱼ = M⁻¹ qⱼ, and (3) keeps its form in u = R y with
# H R⁻¹ in place of H. With `flexible=true` the zⱼ come from the Arnoldi iteration (flexible GMRES,
# Saad 1993, which also allows a preconditioner that changes between Arnoldi steps) and are
# orthonormalised as they come, Zₙ = Wₙ R with Wₙ orthonormal and R upper triangular: the step is
# Wₙ R y, and R keeps the precision of the zⱼ however their lengths spread. Otherwise G comes from
# `gram(M, Qₙ)`, which a preconditioner can compute cheaply, and the step is M⁻¹ (Qₙ y). A Gram
# matrix squares the spread of the lengths, wide when M⁻¹ solves a few slow directions exactly; it
# is scaled to unit diagonal before its root is taken, but some directions are still lost
# (hookstep.jl), and `flexible=true` keeps them. A preconditioner that can give R without forming
# G returns it from `gram`, and loses nothing either.
#
# Trust-region update. The model predicts the reduction β² - ‖g - H y‖² of ‖r‖², and the step is
# judged by the ratio ρ of the actual reduction to the predicted one.
#   trust=:ratio   ρ < 1/4 shrinks Δ by 4, ρ > 3/4 on the boundary doubles it up to `Δmax`;
#                  the step is accepted if ρ > `eta`, otherwise the hookstep for the smaller Δ is
#                  computed in the same Krylov space, without new products with J.
#   trust=:scan    a trial costs one residual against the products of the whole Krylov space, so
#                  the hooksteps at the radii Δ p, p = 1/8 to 4 in steps of √2 (capped by `Δmax` and
#                  by the length of the GMRES step), are all tried; the one of smallest residual
#                  among those with ρ > `eta` is accepted, and the next iteration starts from the
#                  length of its step. If none is accepted, the scan starts again below.

export nkroot!, update!, gram

# the factors of the radius tried by trust=:scan, 1/8 to 4 in steps of √2
const SCAN = 2.0 .^ (-3:0.5:2)

"""
    nkroot!(F!, J, x; precond=nothing, flexible=false, krylov_dim=50, krylov_tol=1e-3,
            Δ=1e-2, Δmax=Inf, eta=1e-3, maxiter=20, tol=1e-10, trust=:ratio,
            verbose=true, io=stdout, callback=info -> false) -> (; status, iter, res)

Solve `F(x) = 0` by the Newton–Krylov hookstep from the guess `x`, overwritten with the root.

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
- `Δ`, `Δmax`: initial and largest trust-region radius. `eta`: smallest ratio of the actual to
  the predicted reduction of `‖r‖²` for a step to be accepted.
- `trust=:ratio` updates the radius from the ratio of each trial; `trust=:scan` tries several radii
  in the same Krylov space and keeps the best.
- The search stops when `‖r‖ < tol`, after `maxiter` iterations, when the trust region collapses,
  or when `callback(info)` returns `true`, called at the start and after every iteration with
  `info = (; iter, x, res, evaluations, krylov, step, radius, steplength, ratio, rejected, trials)`:
  the cumulative counts of residuals, Jacobian products and preconditioner applications, the
  relative residual of the GMRES solution after each Arnoldi step, `"newton"` or `"hook"`, the
  radius of the accepted step, its length, its ratio, the number of rejected trials and the radius
  and ratio of every trial but the accepted one. A [`Trace`](@ref) records them.

The vectors must broadcast and support `similar`, `dot` and `norm`.
"""
function nkroot!(F!, J, x; precond=nothing,
                           flexible::Bool=false,
                           krylov_dim::Int=50,
                           krylov_tol::Real=1e-3,
                           Δ::Real=1e-2,
                           Δmax::Real=Inf,
                           eta::Real=1e-3,
                           maxiter::Int=20,
                           tol::Real=1e-10,
                           trust::Symbol=:ratio,
                           verbose::Bool=true,
                           io::IO=stdout,
                           callback=info -> false)

    # ---- input checks ----
    trust in (:ratio, :scan) ||
        throw(ArgumentError("trust must be :ratio or :scan"))

    # ---- the first radius obeys the largest one, as all the others do ----
    Δ = min(Δ, Δmax)

    # ---- evaluations, counted ----
    counts         = [0, 0, 0] # residuals, Jacobian products, preconditioner applications
    residual!(o, v)     = (counts[1] += 1; F!(o, v))
    product!(o, v)      = (counts[2] += 1; mul!(o, J, v))
    precondition!(o, v) = (counts[3] += 1; ldiv!(o, precond, v))
    evaluations()       = (residuals=counts[1], products=counts[2], preconditioner=counts[3])

    # ---- workspace ----
    r  = residual!(similar(x), x) # residual at x
    rn = similar(x)        # residual at the trial point
    b  = similar(x)        # right-hand side of (1)
    z  = similar(x)        # preconditioned step Qₙ y
    w  = similar(x)        # M⁻¹ v inside the Arnoldi operator
    δx = similar(x)        # step
    xn = similar(x)        # trial point x + δx
    xb = similar(x)        # best trial point of a scan
    rb = similar(x)        # and its residual
    zw = similar(x)        # M⁻¹ v of flexible GMRES
    Q  = _basis(x)         # the Arnoldi vectors, kept from one iteration to the next
    W  = _basis(x)         # flexible: the vectors zⱼ = M⁻¹ qⱼ orthonormalised, Zₙ = Wₙ R
    ϵ  = 100 * eps(Float64) # relative singular value of R, or eigenvalue of G, set by rounding

    res    = norm(r)
    status = :maxiter
    iter   = 0

    verbose && _print_header(io)
    verbose && _print_row(io, 0, res, Δ, NaN, 0, NaN, "", 0)
    stop = callback((; iter, x, res, evaluations=evaluations(), krylov=Float64[], step="", radius=Δ, steplength=NaN,
                       ratio=NaN, rejected=0, trials=Tuple{Float64, Float64}[]))

    while iter < maxiter
        stop && (status = :stopped; break)
        res < tol && (status = :converged; break)
        iter += 1

        # ---- Newton system (1) at x ----
        update!(J, x, r)
        precond === nothing || update!(precond, x)
        b .= .-r
        β  = norm(b)

        # ---- Krylov space of J M⁻¹, one Arnoldi step at a time ----
        empty!(Q)
        empty!(W)
        Rc = Vector{Float64}[]    # flexible: the columns of R, upper triangular
        A! = precond === nothing ? (out, v) -> product!(out, v) :
             flexible            ? (out, v) -> _flexible!(out, product!, precondition!, v, W, Rc, zw) :
                                   (out, v) -> product!(out, precondition!(w, v))
        arn    = ArnoldiIteration(A!, Q, copy(b))
        g      = [β]
        H      = arn.H
        krylov = Float64[]                       # relative residual of the GMRES solution
        qr     = (c=Float64[], s=Float64[], γ=[β]) # Givens QR of H, one column at a time

        for _ in 1:(β > 0 ? krylov_dim : 0)
            # one more basis vector, one more column of H, one more entry of g
            _, H = step!(arn)
            push!(g, 0.0)

            # GMRES solution in the current space: stop once it solves the Newton system well enough
            push!(krylov, _givens!(qr, H[:, end]) / β)
            krylov[end] < krylov_tol && break

            # breakdown: the new direction lies in the space, which no further step can enlarge
            H[end, end] <= 1e-12 * β && break
        end

        # ---- the hookstep (3) in the final space, in the metric of the step ----
        m      = size(H, 2)
        R      = precond === nothing || !flexible ? nothing :
                 UpperTriangular([i <= j ? Rc[j][i] : 0.0 for i in 1:m, j in 1:m])
        Rm, Ri = precond === nothing ? (I, I) :
                 flexible            ? _root(R, ϵ) : _root(_metric(gram(precond, view(Q, 1:m))), ϵ)
        precond === nothing || flexible || (counts[3] += _gram_applications(precond, m))
        SH     = svd(H * Ri)
        hook(ρ) = (h = _hookstep(SH, g, ρ); (Ri * h[1], h[2]))
        len(y)  = norm(Rm * y)
        move!(out, y) = precond === nothing ? lincomb!(out, Q, y) :
                        flexible            ? lincomb!(out, W, R * y) : precondition!(out, lincomb!(z, Q, y))

        # a trial step of Krylov coordinates y: its point into xn, its residual into rn, the norm of the
        # residual and the ratio of the actual to the predicted reduction of ‖r‖²
        function trial!(xn, rn, y)
            move!(δx, y)
            xn  .= x .+ δx
            resn = norm(residual!(rn, xn))
            return resn, (res^2 - resn^2) / (β^2 - norm(g - H * y)^2)
        end

        # ---- trust region: the accepted trial, by the update rule of `trust` ----
        Δmin = eps() * (1 + norm(x))
        t    = trust === :ratio ? _ratio(trial!, hook, len, Δ, Δmax, eta, Δmin, xn, rn) :
                                  _scan(trial!, hook, len, Δ, Δmax, eta, Δmin, xn, rn, xb, rb)
        Δ    = t.Δ
        t.accepted || (status = :collapsed; break)
        x   .= t.x
        r   .= t.r
        res  = t.res
        rejected = count(q -> !(q[2] > eta), t.trials)

        # ---- output ----
        kind = t.at ? "hook" : "newton"
        verbose && _print_row(io, iter, res, Δ, t.ρ, m, krylov[end], kind, rejected)
        stop = callback((; iter, x, res, evaluations=evaluations(), krylov, step=kind, radius=t.used,
                           steplength=len(t.y), ratio=t.ρ, rejected, trials=t.trials))
    end

    res < tol && (status = :converged)
    verbose && _print_footer(io, status, iter, res, evaluations())

    return (; status, iter, res)
end


# ---------------------------------------------------------------------------- #
# trust-region update rules                                                    #
# ---------------------------------------------------------------------------- #
# Each rule tries hooksteps in the Krylov space of one Newton system, by `trial!(xn, rn, y)`, which
# writes the trial point and its residual and returns the norm of the residual and the ratio ρ,
# until it accepts one. It returns the accepted trial: its coordinates `y`, whether it lies on the
# boundary `at`, its ratio `ρ`, the radius `used` with which it was computed, the radius `Δ` of the
# next iteration, its point `x`, residual `r` and residual norm `res`, and the radius and ratio of
# every other trial; or `accepted = false` when the radius falls below `Δmin`.

# trust=:ratio: from the ratio of each trial, Δ shrinks by 4 below 1/4 and doubles above 3/4 on the
# boundary; the first trial with ρ > eta is accepted, the others give a shorter hookstep. A rejected
# step inside the region is the same for every radius above its length, so Δ falls below that
# length at once, with no new residual of the same point. A radius that stays infinite after a
# rejection has no shorter step to give, and the region has collapsed
function _ratio(trial!, hook, len, Δ, Δmax, eta, Δmin, xn, rn)
    trials = Tuple{Float64, Float64}[]
    used   = Δ
    y, at  = hook(Δ)
    while true
        res, ρ = trial!(xn, rn, y)

        # radius update; a ratio that is not a number counts as a poor step, and a rejected trial
        # shrinks the radius whatever eta is, or the same trial would come back
        if !(ρ >= 1/4) || !(ρ > eta)
            Δ /= 4
        elseif ρ > 3/4 && at
            Δ = min(2Δ, Δmax)
        end

        # accepted, or a shorter hookstep in the same space unless the region has collapsed
        ρ > eta && return (accepted=true, y=y, at=at, ρ=ρ, used=used, Δ=Δ, x=xn, r=rn, res=res, trials=trials)
        push!(trials, (used, ρ))
        if !at
            ℓ = len(y)
            isinf(Δ) && (Δ = ℓ)
            while Δ >= max(ℓ, Δmin)
                Δ /= 4
            end
        end
        Δmin <= Δ < Inf || return (accepted=false, y=y, at=at, ρ=ρ, used=used, Δ=Δ, x=xn, r=rn, res=res, trials=trials)
        y, at = hook(Δ)
        used  = Δ
    end
end

# trust=:scan: the hooksteps at the radii Δ p for the factors p of SCAN, capped by Δmax and by the
# length of the GMRES step, which every larger radius gives; the trial of smallest residual among
# those with ρ > eta is accepted, and the next radius is the length of its step. If none has
# ρ > eta, the scan starts again below the smallest radius tried
function _scan(trial!, hook, len, Δ, Δmax, eta, Δmin, xn, rn, xb, rb)
    yN, _ = hook(Inf)
    ℓN    = isfinite(len(yN)) ? len(yN) : Inf
    tried = Tuple{Float64, Float64}[] # radius and ratio of every trial
    while true
        best = 0          # the best trial of this scan
        resb = Inf        # its residual norm
        yb   = yN         # its coordinates
        atb  = false      # and whether it lies on the boundary
        for ρr in unique(min.(Δ .* SCAN, Δmax, ℓN))
            yr, at = hook(ρr)
            res, q = trial!(xn, rn, yr)
            push!(tried, (ρr, q))
            if q > eta && res < resb
                best = length(tried)
                resb = res
                yb   = yr
                atb  = at
                xb  .= xn
                rb  .= rn
            end
        end

        # the best trial, or a scan below the smallest radius tried unless the region has collapsed
        best > 0 && return (accepted=true, y=yb, at=atb, ρ=tried[best][2], used=tried[best][1],
                            Δ=min(len(yb), Δmax), x=xb, r=rb, res=resb, trials=tried[[1:best - 1; best + 1:end]])
        Δ = minimum(first.(tried)) / 8
        Δmin <= Δ < Inf || return (accepted=false, y=yb, at=atb, ρ=NaN, used=Δ, Δ=Δ, x=xb, r=rb, res=resb, trials=tried)
    end
end

"""
    nkroot!(F!, x; jacobian=:fd, ϵ=nothing, kwargs...)

As `nkroot!(F!, J, x; kwargs...)` with the Jacobian by finite differences of `F!` (`jacobian=:fd`)
or by dual numbers (`jacobian=:ad`). The finite difference takes the step of NITSOL, from the
machine precision and the norm of `x`, unless `ϵ` sets a relative step (see [`jacobianFD`](@ref)).
"""
function nkroot!(F!, x; jacobian::Symbol=:fd, ϵ::Union{Real, Nothing}=nothing, kwargs...)
    # ---- input checks ----
    jacobian in (:fd, :ad) ||
        throw(ArgumentError("jacobian must be :fd or :ad"))

    J = jacobian === :ad ? jacobianAD(F!, x) :
        ϵ === nothing    ? jacobianFD(F!, x) : jacobianFD(F!, x, ϵ; step_size=Average())
    return nkroot!(F!, J, x; kwargs...)
end


# ---------------------------------------------------------------------------- #
# preconditioner                                                               #
# ---------------------------------------------------------------------------- #

"""
    update!(J, x, r)
    update!(M, x)

Linearise the Jacobian `J` at `x`, with `r = F(x)`, and the preconditioner `M` at `x`. The
defaults do nothing, for operators that do not depend on `x`.
"""
update!(J, x, r) = J
update!(M, x)    = M

"""
    gram(M, Q) -> Matrix or UpperTriangular

The Gram matrix `G[i, j] = dot(M⁻¹ Q[i], M⁻¹ Q[j])` of the orthonormal Krylov vectors `Q`, which
gives the length of a step of the preconditioned hookstep, or an `UpperTriangular` `R` with
`Rᵀ R = G`. This default applies `M⁻¹` to each vector and returns `G`; a preconditioner can define
a cheaper method. A Gram matrix loses the short vectors when `M⁻¹` stretches a few directions far.
`R` keeps them if it is computed without forming `G`, and so does `flexible=true`, which does not
call `gram`.
"""
function gram(M, Q)
    Z = [ldiv!(similar(q), M, q) for q in Q]
    return [dot(Z[i], Z[j]) for i in eachindex(Z), j in eachindex(Z)]
end

# what gram gives, as _root takes it
_metric(G::UpperTriangular) = G
_metric(G::AbstractMatrix)  = Symmetric(G)

# The Arnoldi operator of flexible GMRES: J z into out, with z = M⁻¹ v. Then z, orthonormalised
# against the vectors before it, goes into W, and its coordinates on them, a column of R, into Rc,
# so that Zₙ = Wₙ R. Gram-Schmidt is applied twice, so that R keeps the precision of the vectors
# when their lengths spread over orders of magnitude, as they do when M⁻¹ solves a few slow
# directions exactly: their inner products would square that spread, and the short vectors would
# fall below rounding. W replaces the vectors zⱼ, with no extra storage, and z is a work vector
function _flexible!(out, product!, precondition!, v, W, Rc, z=similar(v))
    # ---- z = M⁻¹ v, and J z ----
    precondition!(z, v)
    product!(out, z)

    # ---- z orthogonalised against W in place, twice, and normalised ----
    c = zeros(length(W) + 1)
    _orthogonalise!(z, W, view(c, 1:length(W)))
    c[end] = norm(z)
    c[end] > 0 && (z ./= c[end])

    _push!(W, z)
    push!(Rc, c)

    return out
end

# the applications of M⁻¹ made by gram: m for the default, none for a preconditioner with its own
_gram_applications(M, m) = which(gram, Tuple{typeof(M), Any}).module === NKRoots ? m : 0


# ---------------------------------------------------------------------------- #
# output                                                                       #
# ---------------------------------------------------------------------------- #
# One row per Newton step. ‖F‖: residual norm. Δ: trust-region radius after the update. ρ: ratio of
# the actual to the predicted reduction. dim K: dimension of the Krylov space. GMRES res.: relative
# residual of the GMRES solution of the Newton system, the full step before the trust region. step: ↪ hook on the boundary of the trust
# region, → newton the full step inside it. rej.: trial steps rejected before the accepted one. The
# last line says how the search ended, ✗ trust region collapsed when no step reduces ‖F‖.
const _ROWS  = ("iter", "‖F‖", "Δ", "ρ", "dim K", "GMRES res.", "step", "rej.")
const _WIDTH = (4, 10, 8, 7, 5, 10, 8, 4)

# a rule of the table, from its left, middle and right corners
_rule(l, m, r, widths) = l * join(("─"^(w + 2) for w in widths), m) * r

# a row of the table, each cell right-aligned
_cells(cells, widths) = "│ " * join((lpad(c, w) for (c, w) in zip(cells, widths)), " │ ") * " │"

function _print_header(io::IO)
    println(io, _rule("╭", "┬", "╮", _WIDTH))
    println(io, _cells(_ROWS, _WIDTH))
    println(io, _rule("├", "┼", "┤", _WIDTH))
end

function _print_row(io::IO, iter, res, Δ, ρ, k, lin, step, rejected)
    kind  = step == "hook" ? "↪ hook" : step == "newton" ? "→ newton" : ""
    first = iter == 0
    cells = (string(iter),
             @sprintf("%.4e", res),
             @sprintf("%.2e", Δ),
             first ? "" : @sprintf("%.3f", ρ),
             first ? "" : string(k),
             first ? "" : @sprintf("%.2e", lin),
             kind,
             first ? "" : string(rejected))
    println(io, _cells(cells, _WIDTH))
    flush(io)
end

# the bottom of the table, and how the search ended
function _print_footer(io::IO, status, iter, res, counts)
    mark = status === :converged ? "✓ converged" :
           status === :collapsed ? "✗ trust region collapsed" :
           status === :maxiter   ? "✗ maxiter reached" : "■ stopped by the callback"
    println(io, _rule("╰", "┴", "╯", _WIDTH))
    @printf(io, "  %s: %d steps, ‖F‖ = %.3e, %d residuals, %d Jacobian products\n",
            mark, iter, res, counts.residuals, counts.products)
    flush(io)
end
