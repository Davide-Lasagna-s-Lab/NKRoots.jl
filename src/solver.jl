using LinearAlgebra
using Printf
using GMRES

export nkroot!

# interface
# x should
# - have similar
# - have a norm
# - be broadcastable


# TODO: modify interface to allow passing of jacobian
function nkroot!(F, x, opts::Options=Options(); callback=nothing)
    # construct jacobian
    if opts.use_AD 
        jac = jacobianAD(F, x)
    else
        jac = jacobianFD(F, x, opts.ϵ)
    end

    # correction
    dx = similar(x)

    # residual
    r = F(similar(x), x)

    # maybe new solution and maybe new residual
    xx = similar(x)
    rr = similar(x)

    # precompute residual norm
    r_norm  = norm(r)

    # initialise trust region
    Δ = opts.Δ_init

    # initialize status
    status       = :maxiter_reached
    it           = opts.maxiter
    gmres_maxit  = 0
    gmres_minres = 1.0

    # print header and zero iteration
    if opts.verbose
        display_header()
        #             (iter, gmres_iter, gmres_r_norm, dx_norm, Δ, ρ, r_norm)
        display_status(   0,          0,            0,       0, Δ, 0, r_norm)
    end

    # execute callback if needed
    if !isnothing(callback)
        callback(0, x)
    end

    for iter = 1:opts.maxiter
        # set initial guess to zero
        dx .= zero(dx)

        # overwrites dx with solution
        if opts.use_hookstep
            _, gmres_r_norm, gmres_iter = gmres!(dx, jac, r, Δ; rel_rtol=opts.gmres_rel_rtol,
                                                                 maxiter=opts.gmres_maxiter,
                                                                       m=opts.gmres_m,
                                                                 verbose=opts.gmres_verbose)
        else
            _, gmres_r_norm, gmres_iter = gmres!(dx, jac, r; rel_rtol=opts.gmres_rel_rtol,
                                                              maxiter=opts.gmres_maxiter,
                                                                    m=opts.gmres_m,
                                                              verbose=opts.gmres_verbose)
        end

        # update GMRES status
        gmres_maxit < gmres_iter ? gmres_maxit = gmres_iter : nothing
        gmres_minres > gmres_r_norm ? gmres_minres = gmres_r_norm : nothing

        # relative erro norm
        gmres_rel_rnorm = gmres_r_norm/r_norm

        # whether the correction is within the trust region boundary
        hits_boundary = norm(dx) ≥ Δ

        # new candiate solution. Note we used the positive residual for the linear system
        xx .= x .- dx

        # new residual
        F(rr, xx)

        # new residual norm
        rr_norm = norm(rr)

        # calculate ratio of actual and predicted reduction of residual norm
        ρ = (r_norm^2 - rr_norm^2)/(r_norm^2 - gmres_r_norm.^2)

        # trust region update
        if ρ  < 1/4
            Δ *= 1/4
        elseif ρ > 3/4 && hits_boundary
            Δ = min(2*Δ, opts.Δ_max)
        end

        # if there is some reduction
        if ρ  > opts.η
            # update solution and residual
            x .= xx
            r .= rr

            # update jacobian (which is hooked onto x)
            update!(jac)
            
            # and residual norm, so we do not need to recompute it
            r_norm = rr_norm
        end

        # correction norm
        dx_norm = norm(dx)

        # display output
        if opts.verbose && (iter % opts.skipiter == 0)
            display_status(iter, gmres_iter, gmres_rel_rnorm, dx_norm, Δ, ρ, r_norm)
        end

        # execute callback if needed
        if !isnothing(callback)
            callback(iter, x)
        end

        # tolerances reached
        if r_norm <  opts.r_norm_tol
            status = :converged
            it = iter
            break
        end
        if dx_norm < opts.dx_norm_tol
            status = :converged
            it = iter
            break
        end
        # if step < opts.min_step
        #     status = :min_step_reached
        #     break
        # end
    end

    
    return status, it, gmres_maxit, gmres_minres
end


# function for printing the iteration status
function display_header()
    header = "+-------+----------+----------------+----------+----------+-----------+----------+\n"*
             "| iter  | gmres_it | gmres_res_norm |  ||dx||  |     Δ    |     ρ     |  ||r||   |\n"*
             "+-------+----------+----------------+----------+----------+-----------+----------+"
    println(header)
    flush(stdout)
    return nothing
end

function display_status(iter, gmres_iter, gmres_r_norm, dx_norm, Δ, ρ, r_norm)
    str = @sprintf "|  %4d | %8d |   %6.4e   | %5.2e | %5.2e | %+5.2e | %5.2e |" iter gmres_iter gmres_r_norm dx_norm Δ ρ r_norm
    println(str)
    flush(stdout)
    return nothing
end
