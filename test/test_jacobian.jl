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
