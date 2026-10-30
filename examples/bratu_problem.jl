# The Bratu problem u'' + exp(u) = 0 with u(0) = u(1) = 0, by second differences on n points, and the
# matrix of the second differences.
const n = 200
const h = 1/(n + 1)

function bratu!(out, u)
    for i in 1:n
        left   = i == 1 ? zero(u[i]) : u[i - 1]
        right  = i == n ? zero(u[i]) : u[i + 1]
        out[i] = (left - 2u[i] + right)/h^2 + exp(u[i])
    end
    return out
end

const A = Tridiagonal(fill(1/h^2, n - 1), fill(-2/h^2, n), fill(1/h^2, n - 1))
