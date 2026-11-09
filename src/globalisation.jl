# The globalisations of nkroot!: the rules that turn the linear model of one Newton system into an
# accepted step.
#
#   globalisation=:hookstep, trust=:classical   the hookstep, with the classical rule of the radius
#   globalisation=:hookstep, trust=:scan        the hookstep at several radii, in the same Krylov space
#   globalisation=:backtracking                 the GMRES step, halved until the residual falls enough
#   globalisation=:none                         the GMRES step, classical Newton
#
# Each rule tries steps of the model by `trial!(y)`, which writes the trial point and its residual
# into ws.xn and ws.rn and returns the norm of the residual and the ratio ρ of the actual to the
# predicted reduction of ‖r‖², until it accepts one with ρ > eta (classical Newton accepts any step
# of finite residual). It returns the accepted trial: its coordinates `y`, its `kind`, "newton" for
# the GMRES step, "hook" for a hookstep on the boundary of the trust region and "line" for a GMRES
# step shortened by backtracking, its ratio `ρ`, the radius `used` with which it was computed (its
# length without a trust region), the fraction `λ` of the GMRES step that it is (NaN for a
# hookstep), the radius `Δ` of the next iteration, its point `x`, residual `r` and residual norm
# `res`, and the radius and ratio of every other trial; or `accepted = false` when the step falls
# below `Δmin`, the rounding of x, or the residual of classical Newton is not finite.

# the factors of the radius tried by trust=:scan, 1/8 to 8 in steps of √2
const SCAN = 2.0 .^ (-3:0.5:3)

# the accepted step of the model, by the rule of globalisation and, for the hookstep, of trust and scan
function _step(trial!, model, ws, Δ; globalisation, trust, scan, Δmax, eta, Δmin)
    globalisation === :none &&
        return _full(trial!, model, ws, Δ)
    globalisation === :backtracking &&
        return _backtracking(trial!, model, ws, Δ; Δmax=Δmax, eta=eta, Δmin=Δmin)
    trust === :classical &&
        return _classical(trial!, model, ws, Δ; Δmax=Δmax, eta=eta, Δmin=Δmin)

    return _scan(trial!, model, ws, Δ; Δmax=Δmax, eta=eta, Δmin=Δmin, pick=scan)
end


# ---------------------------------------------------------------------------- #
# the hookstep                                                                 #
# ---------------------------------------------------------------------------- #

# trust=:classical: from the ratio of each trial, Δ shrinks by 4 below 1/4 and doubles above 3/4 on the
# boundary; the first trial with ρ > eta is accepted, the others give a shorter hookstep. A rejected
# step inside the region is the same for every radius above its length, so Δ falls below that
# length at once, with no new residual of the same point. A radius that stays infinite after a
# rejection has no shorter step to give, and the region has collapsed
function _classical(trial!, model, ws, Δ; Δmax, eta, Δmin)
    trials = Tuple{Float64, Float64}[]
    used   = Δ
    y, at  = model.hook(Δ)
    while true
        res, ρ = trial!(y)

        # ---- the radius: a ratio that is not a number counts as a poor step, and a rejected trial
        #      shrinks the radius whatever eta is, or the same trial would come back ----
        if !(ρ >= 1/4) || !(ρ > eta)
            Δ /= 4
        elseif ρ > 3/4 && at
            Δ = min(2Δ, Δmax)
        end

        # ---- accepted, or a shorter hookstep in the same space unless the region has collapsed ----
        accepted = ρ > eta
        accepted || push!(trials, (used, ρ))
        if !accepted && !at
            ℓ = model.len(y)
            isinf(Δ) && (Δ = ℓ)
            while Δ >= max(ℓ, Δmin)
                Δ /= 4
            end
        end
        (accepted || !(Δmin <= Δ < Inf)) &&
            return (accepted = accepted,
                    y        = y,
                    kind     = at ? "hook" : "newton",
                    ρ        = ρ,
                    used     = used,
                    λ        = NaN,
                    Δ        = Δ,
                    x        = ws.xn,
                    r        = ws.rn,
                    res      = res,
                    trials   = trials)
        y, at = model.hook(Δ)
        used  = Δ
    end
end

# trust=:scan: the hooksteps at the radii Δ p for the factors p of SCAN, capped by Δmax and by the
# length of the GMRES step, which every larger radius gives. Among the trials with ρ > eta, a step
# that lowers ‖F‖, the scan accepts the one of smallest residual, pick=:greedy, or the one of
# largest radius, the longest step, pick=:longest. The longest step may lower ‖F‖ less now and keep
# longer steps open after it, as in the delayed gratification of Transtrum, Machta and Sethna (2011,
# section VIII C) for the damping of Levenberg–Marquardt. The next radius is the length of the
# accepted step. If none has ρ > eta, the scan starts again below the smallest radius tried
function _scan(trial!, model, ws, Δ; Δmax, eta, Δmin, pick)
    yN, _ = model.hook(Inf)
    ℓN    = isfinite(model.len(yN)) ? model.len(yN) : Inf
    tried = Tuple{Float64, Float64}[] # radius and ratio of every trial
    while true
        best  = 0          # the best trial of this scan
        resb  = Inf        # its residual norm
        usedb = 0.0        # its radius
        yb    = yN         # its coordinates
        atb   = false      # and whether it lies on the boundary
        for ρr in unique(min.(Δ .* SCAN, Δmax, ℓN))
            yr, at = model.hook(ρr)
            res, q = trial!(yr)
            push!(tried, (ρr, q))

            # a step that lowers ‖F‖, better than the best so far by the rule of pick
            better = pick === :greedy ? res < resb : ρr > usedb
            if q > eta && better
                best    = length(tried)
                resb    = res
                usedb   = ρr
                yb      = yr
                atb     = at
                ws.xb  .= ws.xn
                ws.rb  .= ws.rn
            end
        end

        # ---- the best trial, or a scan below the smallest radius tried unless the region has
        #      collapsed ----
        best > 0 &&
            return (accepted = true,
                    y        = yb,
                    kind     = atb ? "hook" : "newton",
                    ρ        = tried[best][2],
                    used     = tried[best][1],
                    λ        = NaN,
                    Δ        = min(model.len(yb), Δmax),
                    x        = ws.xb,
                    r        = ws.rb,
                    res      = resb,
                    trials   = tried[[1:best - 1; best + 1:end]])
        Δ = minimum(first.(tried)) / 8
        Δmin <= Δ < Inf ||
            return (accepted = false,
                    y        = yb,
                    kind     = "hook",
                    ρ        = NaN,
                    used     = Δ,
                    λ        = NaN,
                    Δ        = Δ,
                    x        = ws.xb,
                    r        = ws.rb,
                    res      = resb,
                    trials   = tried)
    end
end


# ---------------------------------------------------------------------------- #
# the GMRES step                                                               #
# ---------------------------------------------------------------------------- #

# globalisation=:backtracking: the GMRES step, its length capped by Δmax, halved until ρ > eta. It
# only shortens the Newton step, where the hookstep also turns it towards the steepest descent of
# ‖F‖. The step collapses when its length falls below Δmin. The radius Δ is kept as it is
function _backtracking(trial!, model, ws, Δ; Δmax, eta, Δmin)
    yN, _  = model.hook(Inf)
    ℓ      = model.len(yN)
    λ      = min(1.0, Δmax / ℓ)
    trials = Tuple{Float64, Float64}[]
    while true
        res, ρ   = trial!(λ .* yN)
        accepted = ρ > eta
        accepted || push!(trials, (λ * ℓ, ρ))

        # ---- accepted, or half the step unless it has collapsed ----
        (accepted || λ * ℓ / 2 < Δmin) &&
            return (accepted = accepted,
                    y        = λ .* yN,
                    kind     = λ == 1 ? "newton" : "line",
                    ρ        = ρ,
                    used     = λ * ℓ,
                    λ        = λ,
                    Δ        = Δ,
                    x        = ws.xn,
                    r        = ws.rn,
                    res      = res,
                    trials   = trials)
        λ /= 2
    end
end

# globalisation=:none: the GMRES step, whatever its residual, as classical Newton takes it; a
# residual that is not finite ends the search. The radius Δ is kept as it is
function _full(trial!, model, ws, Δ)
    yN, _  = model.hook(Inf)
    res, ρ = trial!(yN)

    return (accepted = isfinite(res),
            y        = yN,
            kind     = "newton",
            ρ        = ρ,
            used     = model.len(yN),
            λ        = 1.0,
            Δ        = Δ,
            x        = ws.xn,
            r        = ws.rn,
            res      = res,
            trials   = Tuple{Float64, Float64}[])
end
