using Parameters

export Options

@with_kw struct Options
    # generic parameters
          maxiter::Int     = 10           # maximum newton iteration number
         skipiter::Int     = 1            # skip iteration between displays
          verbose::Bool    = true         # print iteration status
      dx_norm_tol::Float64 = 1e-10        # tolerance on correction
       r_norm_tol::Float64 = 1e-10        # tolerance on residual
                ϵ::Float64 = 1e-6         # step for finite difference approximation
           use_AD::Bool    = false        # use automatic differentiation?

    # GMRES parameters
    gmres_maxiter::Int     = 10           # maximum number of GMRES iterations
    gmres_verbose::Bool    = false        # print GMRES iteration status
   gmres_rel_rtol::Float64 = 1e-2         # GMRES relative stopping tolerance
          gmres_m::Int     = 10           # restart parameter

    # hookstep algorithm parameters
     use_hookstep::Bool    = true         # whether to use hookstep in GMRES
         min_step::Float64 = 1e-4         # minimum step tolerance
                α::Float64 = 1
           NR_lim::Float64 = 1e-8
           Δ_init::Float64 = 1            # initial trust region radius
            Δ_max::Float64 = 10^8         # maximum trust region radius
            Δ_min::Float64 = 10^(-8)      # maximum trust region radius
                η::Float64 = 0.00         # maximum trust region radius
end
