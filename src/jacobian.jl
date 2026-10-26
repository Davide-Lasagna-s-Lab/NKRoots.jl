export jacobianFD,
       jacobianAD

using DualNumbers

# Object to compute the action of the jacobian of a function exactly, by dual numbers.
# Assumes F obeys F(out, x) and overwrites `out` with F(x), returning out.
struct JacobianAD{X, FT, TUP}
      F::FT  # operator
      x::X   # point at which the derivative is taken
    tmp::TUP # temporaries with dual
end

# outer constructor
jacobianAD(F, x) =
    JacobianAD(F, copy(x), (similar(x, Dual{eltype(x)}), similar(x, Dual{eltype(x)})))

# linearise at x
update!(J::JacobianAD, x, r) = (J.x .= x; J)

# jacobian-vector product
function LinearAlgebra.mul!(out, op::JacobianAD, y)
    # aliases
     x_dual = op.tmp[1]
    Fx_dual = op.tmp[2]

    # perturbed input
    x_dual .= op.x .+ DualNumbers.ε .* y

    # calculate function with perturbed input
    op.F(Fx_dual, x_dual)

    # exact gradient
    out .= dualpart.(Fx_dual)

    return out
end

Base.:*(op::JacobianAD, y) = mul!(similar(y), op, y)

# type interface to allow dispatch on method to determine FD step size
abstract type EpsilonMethod end
struct Average <: EpsilonMethod end
struct Nitsol <: EpsilonMethod end

# Object to approximate the action of the jacobian of a function
# using a finite difference quotient.
# Assumes F obeys :
# F(out, x)
# and overwrites `out` with F(x), returning out
struct JacobianFD{X, FT, STEP}
      F::FT          # operator
      x::X           # point at which the derivative is taken
     Fx::X           # value of the function at `x`
    tmp::Tuple{X, X} # temporaries
      ϵ::Float64     # finite difference step
end

"""
    jacobianFD(F, x, ϵ=1e-6; step_size=Nitsol())

The Jacobian of `F` at `x` by a forward difference quotient, `J y ≈ (F(x + δ y) - F(x))/δ`,
with δ from `step_size` (see page 363 of
https://www.sciencedirect.com/science/article/pii/S0021999103004340). `update!(J, x, r)` moves it
to `x`, with `r = F(x)`, without a new evaluation of `F`.
"""
jacobianFD(F, x, ϵ::Real=1e-6; step_size::EpsilonMethod=Nitsol()) =
    JacobianFD{typeof(x), typeof(F), typeof(step_size)}(F, copy(x), F(similar(x), x), (similar(x), similar(x)), ϵ)

_get_step(op::JacobianFD{<:Any, <:Any, Average}, y) = (1/(length(op.x)*norm(y)))*sum(op.ϵ.*abs.(op.x)) + op.ϵ
_get_step(op::JacobianFD{<:Any, <:Any, Nitsol}, y) = sqrt((1 + norm(op.x))*eps(Float64))/norm(y)

# linearise at x, with r = F(x)
update!(J::JacobianFD, x, r) = (J.x .= x; J.Fx .= r; J)

# jacobian-vector product
function LinearAlgebra.mul!(out, op::JacobianFD, y)
    # aliases
     x_plus_y = op.tmp[1]
    Fx_plus_y = op.tmp[2]

    # if perturbation is zero set output to zero
    if norm(y) == 0
        out .= zero(out)
        return out
    end

    # perturbed input, and the finite difference quotient
    δ = _get_step(op, y)
    x_plus_y .= op.x .+ δ .* y
    op.F(Fx_plus_y, x_plus_y)
    out .= (Fx_plus_y .- op.Fx) ./ δ

    return out
end

Base.:*(op::JacobianFD, y) = mul!(similar(y), op, y)
