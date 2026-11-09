# The trust-region update rules of nkroot!.
#
# Each rule tries hooksteps in the Krylov space of one Newton system, by `trial!(xn, rn, y)`, which
# writes the trial point and its residual and returns the norm of the residual and the ratio ρ,
# until it accepts one. It returns the accepted trial: its coordinates `y`, whether it lies on the
# boundary `at`, its ratio `ρ`, the radius `used` with which it was computed, the radius `Δ` of the
# next iteration, its point `x`, residual `r` and residual norm `res`, and the radius and ratio of
# every other trial; or `accepted = false` when the radius falls below `Δmin`.

# the factors of the radius tried by trust=:scan, 1/8 to 4 in steps of √2
const SCAN = 2.0 .^ (-3:0.5:2)

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
# length of the GMRES step, which every larger radius gives. Among the trials with ρ > eta, a step
# that lowers ‖F‖, the scan accepts the one of smallest residual, pick=:residual, or the one of
# largest radius, the longest step, pick=:longest. The longest step may lower ‖F‖ less now and keep
# longer steps open after it, as in the delayed gratification of Transtrum, Machta and Sethna (2011,
# section VIII C) for the damping of Levenberg–Marquardt. The next radius is the length of the
# accepted step. If none has ρ > eta, the scan starts again below the smallest radius tried
function _scan(trial!, hook, len, Δ, Δmax, eta, Δmin, xn, rn, xb, rb, pick)
    yN, _ = hook(Inf)
    ℓN    = isfinite(len(yN)) ? len(yN) : Inf
    tried = Tuple{Float64, Float64}[] # radius and ratio of every trial
    while true
        best  = 0          # the best trial of this scan
        resb  = Inf        # its residual norm
        usedb = 0.0        # its radius
        yb    = yN         # its coordinates
        atb   = false      # and whether it lies on the boundary
        for ρr in unique(min.(Δ .* SCAN, Δmax, ℓN))
            yr, at = hook(ρr)
            res, q = trial!(xn, rn, yr)
            push!(tried, (ρr, q))

            # a step that lowers ‖F‖, better than the best so far by the rule of pick
            better = pick === :residual ? res < resb : ρr > usedb
            if q > eta && better
                best  = length(tried)
                resb  = res
                usedb = ρr
                yb    = yr
                atb   = at
                xb   .= xn
                rb   .= rn
            end
        end

        # the best trial, or a scan below the smallest radius tried unless the region has collapsed
        best > 0 && return (accepted=true, y=yb, at=atb, ρ=tried[best][2], used=tried[best][1],
                            Δ=min(len(yb), Δmax), x=xb, r=rb, res=resb, trials=tried[[1:best - 1; best + 1:end]])
        Δ = minimum(first.(tried)) / 8
        Δmin <= Δ < Inf || return (accepted=false, y=yb, at=atb, ρ=NaN, used=Δ, Δ=Δ, x=xb, r=rb, res=resb, trials=tried)
    end
end
