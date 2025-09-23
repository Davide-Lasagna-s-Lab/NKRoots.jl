export jacobianFD,
       jacobianAD,
       update!

using DualNumbers
 
struct JacobianAD{X, FT, TUP}
    F::FT  # operator
    x::X   # point at which the derivative is taken
  tmp::TUP # temporaries with dual
end

# outer constructor
jacobianAD(F, x) = 
    JacobianAD(F, x, (similar(x, Dual{eltype(x)}), similar(x, Dual{eltype(x)})))

# does nothing
update!(::JacobianAD) = nothing

# matrix vector product function required by GMRES
function Base.:*(op::JacobianAD{X}, y::X) where {X}
    # create output
    out = similar(y)

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

# outer constructor
jacobianFD(F, x, ϵ::Float64=1e-6; step_size::EpsilonMethod=Nitsol()) =
    JacobianFD{typeof(x), typeof(F), typeof(step_size)}(F, x, F(similar(x), x), (similar(x), similar(x)), ϵ)

# methods to compute FD step size (see page 363 of https://www.sciencedirect.com/science/article/pii/S0021999103004340)
_get_step(op::JacobianFD{<:Any, <:Any, Average}, y) = (1/(length(op.x)*norm(y)))*sum(op.ϵ.*abs.(op.x)) + op.ϵ
_get_step(op::JacobianFD{<:Any, <:Any, Nitsol}, y) = sqrt((1 + norm(op.x))*eps(Float64))/norm(y)

# update Fx assuming x has been modified outside
function update!(j::JacobianFD{X}) where {X}
    j.F(j.Fx, j.x)
    return nothing
end

# matrix vector product function required by GMRES
function Base.:*(op::JacobianFD{X, <:Any, STEP}, y::X) where {X, STEP}
    # create output
    out = similar(y)

    # aliases
     x_plus_y = op.tmp[1]
    Fx_plus_y = op.tmp[2]
            x = op.x
           Fx = op.Fx
            δ = _get_step(op, y)

    # if perturbation is zero set output to zero
    if norm(y) == 0
        out .= zero(out)
    else
        # perturbed input
        x_plus_y .= x .+ δ .* y
        
        # calculate function with perturbed input
        op.F(Fx_plus_y, x_plus_y)

        # calculate finite difference approximation of gradient 
        out .= (Fx_plus_y .- Fx)./δ
    end

    return out
end
