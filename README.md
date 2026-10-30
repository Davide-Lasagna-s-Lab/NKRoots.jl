# NKRoots.jl

A Newton–Krylov hookstep solver for `F(x) = 0` in Julia. It needs only the action of the Jacobian
on a vector, so it suits large systems, and it works on any vector type that broadcasts.

- The step comes from a trust region in a Krylov space, the hookstep of Viswanath (2007).
- A rejected step costs one residual and no new Krylov space.
- The Jacobian can be a finite difference, a dual-number derivative, or your own operator.
- A right preconditioner is optional. The trust region stays on the step in `x`.
- The whole history of a search can be recorded with `Trace`.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/NKRoots.jl")
```

## A first example

Two equations in two unknowns. The function writes the residual into `out` and returns it. The
guess `x` is overwritten with the root.

```julia
using NKRoots

function F!(out, x)
    out[1] = x[1]^2 + x[2]^2 - 4
    out[2] = exp(x[1]) + x[2] - 1
    return out
end

x   = [1.0, -1.0]
out = nkroot!(F!, x; Δ=1.0, tol=1e-12)
```

The Jacobian comes from a finite difference. Each row of the output is one Newton step.

```
  iter      ‖r‖        radius       ρ      krylov    linear    step   rejected
     0  2.1251e+00  1.000e+00       NaN       0  NaN
     1  8.5999e-01  1.000e+00     0.836       2  1.96e-32  newton         0
     2  3.9640e-02  1.000e+00     0.998       2  4.96e-32  newton         0
     3  1.0240e-04  1.000e+00     1.000       2  1.53e-32  newton         0
     4  8.4147e-10  1.000e+00     1.000       2  0.00e+00  newton         0
     5  0.0000e+00  1.000e+00     1.000       2  3.45e-32  newton         0
```

Here `out` is `(status = :converged, iter = 5, res = 0.0)` and `x` is
`[1.0041687384746592, -1.72963728702587]`. The output columns are the residual norm, the radius of
the trust region, the ratio of the actual to the predicted reduction, the size of the Krylov space,
the relative residual of the GMRES solution, the kind of step and the rejected trials. The step is
`newton` inside the trust region and `hook` on its boundary.

## An exact derivative

With `jacobian=:ad` the product with the Jacobian comes from dual numbers. The function must then
accept vectors of dual numbers, as `F!` above does.

```julia
x   = [1.0, -1.0]
out = nkroot!(F!, x; jacobian=:ad, Δ=1.0, tol=1e-12, verbose=false)
```

## A preconditioner

The Bratu problem `u'' + exp(u) = 0` with `u(0) = u(1) = 0`, on 200 points. The Laplacian makes the
Newton system stiff. Its LU factorisation is a good right preconditioner, and any object with
`ldiv!(out, M, v)` will do.

```julia
using LinearAlgebra

n = 200
h = 1/(n + 1)

function bratu!(out, u)
    for i in 1:n
        left   = i == 1 ? zero(u[i]) : u[i - 1]
        right  = i == n ? zero(u[i]) : u[i + 1]
        out[i] = (left - 2u[i] + right)/h^2 + exp(u[i])
    end
    return out
end

A = Tridiagonal(fill(1/h^2, n - 1), fill(-2/h^2, n), fill(1/h^2, n - 1))
M = lu(A)

trace = Trace()
u     = zeros(n)
out   = nkroot!(bratu!, u; precond=M, Δ=1.0, krylov_tol=1e-6, tol=1e-8, verbose=false, callback=trace)

trace.res           # residual norms, in order
trace.products[end] # Jacobian products of the whole search
```

Both searches below take four Newton steps and reach the same `u`, to `3e-15`. The preconditioner
changes the cost, not the root.

| | Jacobian products |
|---|---:|
| no preconditioner, `krylov_dim=250` | 561 |
| `precond=lu(A)` | 12 |

`Trace` also keeps the wall time, the counts of residuals and preconditioner applications, the
GMRES residual after each Arnoldi step, the radius, the step length, the ratio and every rejected
trial.

Use `flexible=true` with a preconditioner that solves a few directions exactly. Such a
preconditioner stretches some vectors far beyond the others, and the length of a step then comes
from flexible GMRES, not from a Gram matrix.

## Your own Jacobian

The Jacobian is any object with `mul!(out, J, v)`. `update!(J, x, r)` linearises it at `x`, with
`r = F(x)`, at the start of every Newton step. Here is the Jacobian of the Bratu problem, without
finite differences. It starts from any point, since the first `update!` sets it.

```julia
struct BratuJacobian
    d::Vector{Float64} # exp(u), the diagonal of the nonlinear term
end

function NKRoots.update!(J::BratuJacobian, u, r)
    J.d .= exp.(u)
    return J
end

function LinearAlgebra.mul!(out, J::BratuJacobian, v)
    for i in 1:n
        left   = i == 1 ? 0.0 : v[i - 1]
        right  = i == n ? 0.0 : v[i + 1]
        out[i] = (left - 2v[i] + right)/h^2 + J.d[i]*v[i]
    end
    return out
end

u   = zeros(n)
out = nkroot!(bratu!, BratuJacobian(zeros(n)), u; precond=M, Δ=1.0, krylov_tol=1e-6, tol=1e-8)
```

A preconditioner that depends on the point defines `update!(M, x)` in the same way.

## How it works

Newton's method solves $J\,\delta x = -r$ with $r = F(x)$. Far from a root the full step is
unreliable, so the step is restricted to a region where the linear model is trusted,

$$\min_{\delta x} \|r + J\,\delta x\| \quad \text{subject to} \quad \|\delta x\| \le \Delta.$$

The Arnoldi iteration builds an orthonormal basis $Q_n$ of the Krylov space from $-r$, with
$J Q_n = Q_{n+1} H$. For $\delta x = Q_n y$ the problem becomes

$$\min_{y} \|\beta e_1 - H y\| \quad \text{subject to} \quad \|y\| \le \Delta,$$

which the SVD of $H$ solves. The Krylov space stops growing when the GMRES residual falls below
`krylov_tol`, or at `krylov_dim`.

With a right preconditioner the Arnoldi iteration runs on $J M^{-1}$ and the step is
$\delta x = M^{-1} Q_n y$. The trust region still bounds $\delta x$, not $y$. Its metric comes from
the vectors $M^{-1} q_j$, or from `gram(M, Q)`, which a preconditioner can compute cheaply.

The ratio $\rho$ of the actual to the predicted reduction of $\|r\|^2$ judges each trial. A poor
trial gives a shorter hookstep in the same Krylov space, with no new products with $J$.

## Keywords

| keyword | default | meaning |
|---|---|---|
| `precond` | `nothing` | right preconditioner, with `ldiv!` |
| `flexible` | `false` | keep the vectors `M⁻¹ qⱼ` for the length of the step |
| `krylov_dim` | `50` | largest dimension of the Krylov space |
| `krylov_tol` | `1e-3` | relative GMRES residual at which the space stops growing |
| `Δ` | `1e-2` | initial radius of the trust region |
| `Δmax` | `Inf` | largest radius |
| `eta` | `1e-3` | smallest ratio for a step to be accepted |
| `maxiter` | `20` | Newton steps at most |
| `tol` | `1e-10` | residual norm of convergence |
| `trust` | `:ratio` | `:ratio` updates the radius after each trial, `:scan` tries several radii in the same space and keeps the best |
| `verbose` | `true` | print one row per step |
| `io` | `stdout` | where to print |
| `callback` | none | called with the state of the search at each step, stops it by returning `true` |

Without a Jacobian, `nkroot!(F!, x; ...)` also takes `jacobian=:fd` or `:ad` and the step `ϵ` of the
finite difference. The search ends with the status `:converged`, `:maxiter`, `:stopped` or
`:collapsed`, the last when no step in the Krylov space reduces the residual.

## References

- D. Viswanath, *Recurrent motions within plane Couette turbulence*, J. Fluid Mech. 580 (2007).
  [arXiv:physics/0604062](https://arxiv.org/abs/physics/0604062).
- D. Viswanath, *The critical layer in pipe flow at high Reynolds number*, Phil. Trans. R. Soc. A
  367 (2009). [arXiv:0809.1498](https://arxiv.org/abs/0809.1498).
- Y. Saad, *A flexible inner-outer preconditioned GMRES algorithm*, SIAM J. Sci. Comput. 14 (1993).
