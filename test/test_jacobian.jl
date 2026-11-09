# simple one-dimensional function
function test(out::Vector, x::Vector)
    out[1] = x[1]^2 - 1
    return out
end

@testset "jacobianFD                               " begin
    # calc derivative at point [2.0]
    x = [2.0]
    J = jacobianFD(test, x)
    @test norm(J*[1.0] - [4.0]) < 1e-5

    # move to x = 4 and check the derivative
    x .*= 2.0
    update!(J, x, test(similar(x), x))
    @test norm(J*[1.0] - [8.0]) < 1e-5

    # the central difference is exact for a quadratic, up to rounding
    J = jacobianFD(test, [2.0]; difference=:central)
    @test norm(J*[1.0] - [4.0]) < 1e-9
    @test norm(J*[0.0]) == 0

    # a relative step sets the step, and an unknown difference is refused
    @test abs((jacobianFD(test, [2.0]; ϵ=1e-2)*[1.0])[1] - 4.0) > 1e-3
    @test_throws ArgumentError jacobianFD(test, [2.0]; difference=:backward)
end

@testset "jacobianAD                               " begin
    # calc derivative at point [2.0]
    x = [2.0]
    J = jacobianAD(test, x)
    @test J*[1.0] == [4.0]

    # move to x = 4 and check the derivative
    x .*= 2.0
    update!(J, x, test(similar(x), x))
    @test J*[1.0] == [8.0]
end
