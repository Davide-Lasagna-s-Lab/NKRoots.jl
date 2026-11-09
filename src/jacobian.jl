export jacobianFD,
       jacobianAD

using DualNumbers

# Object to compute the action of the jacobian of a function exactly, by dual numbers.
# Assumes F obeys F(out, x) and overwrites `out` with F(x), returning out.
struct JacobianAD{X, FT, TUP}
        F::FT                      # operator
        x::X                       # point at which the derivative is taken
      tmp::TUP                     # temporaries with dual
    calls::Base.RefValue{Int}      # calls of F, cumulative
end

# outer constructor
jacobianAD(F, x) =
    JacobianAD(F, copy(x), (similar(x, Dual{eltype(x)}), similar(x, Dual{eltype(x)})), Ref(0))

# the calls of F made by a Jacobian: none for one that NKRoots.jl does not build
_calls(J) = 0
_calls(J::JacobianAD) = J.calls[]

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
    op.calls[] += 1

    # exact gradient
    out .= dualpart.(Fx_dual)

    return out
end

Base.:*(op::JacobianAD, y) = mul!(similar(y), op, y)

# Object to approximate the action of the Jacobian of a function by a finite difference quotient,
# forward or central,
#
#     J y ≈ (F(x + ε y) - F(x))/ε             forward, first order
#     J y ≈ (F(x + ε y) - F(x - ε y))/(2ε)    central, second order
#
# with the step ε = δ (1 + ‖x‖)/‖y‖. By default δ = eps^(1/2) forward and eps^(1/3) central, with eps
# the machine precision of the vectors, each balancing the truncation error of the quotient against
# the rounding of F. For a quadratic F the central difference is exact up to rounding. The residual
# itself is not approximated, so that a root found with this Jacobian satisfies the equations to the
# same tolerance. A forward product evaluates F once, with F(x) set by update! at no cost, a central
# one twice. F obeys F(out, x), and overwrites out with F(x), returning out.
struct JacobianFD{FT, X}
             F::FT      # the function, F(out, x) -> out
             x::X       # the point at which the derivative is taken
            Fx::X       # F at x, for forward differences
           tmp::X                   # the perturbed point x ± ε y
          work::X                   # F at x - ε y, for central differences
    difference::Symbol              # :forward or :central
             δ::Float64             # the relative step
         calls::Base.RefValue{Int}  # calls of F, cumulative, the one at x included
end

_calls(J::JacobianFD) = J.calls[]

"""
    jacobianFD(F, x; difference=:forward, ϵ=nothing)

The Jacobian of `F` at `x` by a finite difference quotient of `F`, matrix free,

    J y ≈ (F(x + ε y) - F(x))/ε               difference=:forward, first order
    J y ≈ (F(x + ε y) - F(x - ε y))/(2ε)      difference=:central, second order

with the step ε = δ (1 + ‖x‖)/‖y‖. The relative step δ is `ϵ` or, by default, the square root of the
machine precision of the vectors for forward differences and its cube root for central ones. A
forward product evaluates `F` once, a central one twice. `update!(J, x, r)` moves it to the point
`x`, with `r = F(x)`, without a new evaluation of `F`.

# Arguments
- `F`: the function, called as `F(out, x)`, which overwrites `out` with `F(x)` and returns it.
- `x`: the point, copied; `F` is evaluated there once.
- `difference::Symbol=:forward`: `:forward` or `:central`.
- `ϵ`: the relative step δ, or `nothing` for the default.
"""
function jacobianFD(F, x; difference::Symbol=:forward, ϵ::Union{Real, Nothing}=nothing)
    # ---- input checks ----
    difference in (:forward, :central) ||
        throw(ArgumentError("difference must be :forward or :central"))

    # ---- the relative step, from the precision of the vectors if they have one ----
    T = eltype(x) <: Number ? real(eltype(x)) : Float64
    δ = ϵ === nothing ? eps(T)^(difference === :forward ? 1/2 : 1/3) : ϵ

    return JacobianFD(F,
                      copy(x),
                      F(similar(x), x),
                      similar(x),
                      similar(x),
                      difference,
                      Float64(δ),
                      Ref(1))
end

# linearise at x, with r = F(x)
update!(J::JacobianFD, x, r) = (J.x .= x; J.Fx .= r; J)

# the product by the difference quotient, written into out
function LinearAlgebra.mul!(out, J::JacobianFD, y)
    ny = norm(y)

    # ---- a zero direction has a zero product ----
    if ny == 0
        out .= 0 .* y
        return out
    end

    # ---- F at x + ε y ----
    ε      = J.δ*(1 + norm(J.x))/ny
    J.tmp .= J.x .+ ε .* y
    J.F(out, J.tmp)
    J.calls[] += 1

    # ---- the forward difference, with F(x) from the point of J ----
    if J.difference === :forward
        out .= (out .- J.Fx) ./ ε
        return out
    end

    # ---- the central difference, with F at x - ε y ----
    J.tmp .= J.x .- ε .* y
    J.F(J.work, J.tmp)
    J.calls[] += 1
    out .= (out .- J.work) ./ (2ε)

    return out
end

Base.:*(J::JacobianFD, y) = mul!(similar(y), J, y)
