# simple one-dimensional function
function test1(out::Vector, x::Vector)
    out[1] = x[1]^2 - 1
    return out
end

@testset "test 1                                   " begin
    for use_AD in [true, false], use_hookstep in [true, false]
        # initial guess
        x = Float64[2.0]

        # default options
        opts = Options(use_AD       = use_AD,
                       use_hookstep = use_hookstep,
                       verbose      = false)

        data = []
        callback(it, x) = push!(data, x[1])

        # run
        status = nkroot!(test1, x, opts; callback=callback)[1]

        # check status
        @test status == :converged

        # check first element
        @test data[1] == 2.0
        @test abs(data[end] - 1.0) < opts.dx_norm_tol

        # somehow it must have exited
        @test ( 
                norm(x - [1.0])       < opts.dx_norm_tol ||
                norm(test1([0.0], x)) < opts.r_norm_tol
            )
    end
end
