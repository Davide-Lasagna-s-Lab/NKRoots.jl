# NKRoots.jl

A Newton–Krylov hookstep solver for `F(x) = 0` in Julia. It needs only the product of the Jacobian
with a vector, so it suits large systems, and `x` can be any type that broadcasts.

The step comes from a trust region in a Krylov space, the hookstep of Viswanath. A rejected step costs
one residual and no new Krylov space. The Jacobian can be a finite difference, dual numbers or your
own operator. A right preconditioner is optional, and the trust region stays on the step in `x`. The
radius follows the classical rule or a scan. `Trace` records everything a search does.

![Routes of four searches on two equations](examples/figures/route.png)

**Examples.** [A first search](#a-first-search) ·
[The Jacobian](#the-jacobian) ·
[The Bratu problem](#the-bratu-problem) ·
[The hookstep](#the-hookstep) ·
[The radius](#the-radius) ·
[The Krylov space](#the-krylov-space) ·
[Preconditioners](#preconditioners) ·
[Tracing](#tracing) ·
[A turbulent channel](#a-turbulent-channel) ·
[Spalart–Allmaras](#spalartallmaras)

**Reference.** [How it works](#how-it-works) · [Keywords](#keywords) · [References](#references)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/NKRoots.jl")
```

The scripts of `examples/` have their own environment, with CairoMakie for the figures.

```
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/route.jl
```

## A first search

Two equations in two unknowns, the circle $x_1^2 + x_2^2 = 2$ and the curve
$e^{x_1 - 1} + x_2^3 = 2$. The function writes the residual into `out` and returns it. The guess `x` is
overwritten with the root.

```julia
using NKRoots

function two!(out, x)
    out[1] = x[1]^2 + x[2]^2 - 2
    out[2] = exp(x[1] - 1) + x[2]^3 - 2
    return out
end

x   = [1.8, 0.7]
out = nkroot!(two!, x; Δ=0.1, tol=1e-12)
```

Each row of the output is one Newton step.

```
  iter         ‖r‖     radius         ρ  krylov    linear    step  rejected
     0  1.8210e+00  1.000e-01       NaN       0       NaN
     1  1.3909e+00  2.000e-01     0.976       2  6.59e-34    hook         0
     2  6.7069e-01  4.000e-01     0.973       2  5.32e-33    hook         0
     3  3.3101e-01  8.000e-01     0.967       2  7.72e-32    hook         0
     4  1.1106e-01  8.000e-01     0.887       2  2.62e-32  newton         0
     5  1.7982e-03  8.000e-01     1.000       2  0.00e+00  newton         0
     6  1.4706e-06  8.000e-01     1.000       2  0.00e+00  newton         0
     7  9.9742e-13  8.000e-01     1.000       2  0.00e+00  newton         0
```

The radius starts small and doubles while the linear model is good. The first three steps sit on
the boundary of the trust region, `hook`. The last four are full Newton steps, `newton`, and the
residual falls quadratically. The columns are the residual norm, the radius, the ratio of the actual
to the predicted reduction, the size of the Krylov space, the relative residual of the GMRES
solution, the kind of step and the rejected trials.

The function returns `(status = :converged, iter = 7, res = 9.97e-13)`, and `x` is the root `(1, 1)`.

The figure at the top shows four searches on the landscape of $\tfrac12\|F\|^2$. The blue and orange
curves are the two equations, and they cross at the two roots, the stars. Three searches reach a
root. The fourth, from `(1.6, -1.2)`, ends in a valley with no root and stops with the status
`:collapsed`. A hookstep lowers $\|F\|$, so it finds a local minimum of $\|F\|$ if the guess leads there.

Script: [`examples/route.jl`](examples/route.jl).

## The Jacobian

There are three ways to give it. A finite difference is the default, `jacobian=:ad` uses dual numbers,
and any object with `mul!` will do. `update!(J, x, r)` moves your own object to the point `x`, with
`r = F(x)`, at the start of every Newton step.

```julia
using LinearAlgebra

x = [1.8, 0.7]
nkroot!(two!, x; jacobian=:fd, Δ=0.1, verbose=false)          # the default
x = [1.8, 0.7]
nkroot!(two!, x; jacobian=:ad, Δ=0.1, verbose=false)          # dual numbers
x = [1.8, 0.7]
nkroot!(two!, x; jacobian=:fd, ϵ=1e-2, Δ=0.1, verbose=false)  # a relative step for the difference

struct TwoJacobian
    A::Matrix{Float64} # the matrix at the last point
end

function NKRoots.update!(J::TwoJacobian, x, r)
    J.A[1, 1] = 2x[1]
    J.A[1, 2] = 2x[2]
    J.A[2, 1] = exp(x[1] - 1)
    J.A[2, 2] = 3x[2]^2
    return J
end

LinearAlgebra.mul!(out, J::TwoJacobian, v) = mul!(out, J.A, v)

x = [1.8, 0.7]
nkroot!(two!, TwoJacobian(zeros(2, 2)), x; Δ=0.1, verbose=false)
```

The three exact Jacobians follow the same path. A coarse finite difference still converges, because
the residual is exact, but it loses the fast final phase of Newton's method.

![Convergence with four Jacobians](examples/figures/jacobians.png)

Script: [`examples/jacobians.jl`](examples/jacobians.jl).

## The Bratu problem

The equation $u'' + e^u = 0$ with $u(0) = u(1) = 0$ has two solutions, and the guess decides which one
a search finds. By second differences on 200 points the residual is

```julia
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

low  = zeros(n)
high = [6*sin(π*i*h) for i in 1:n]
nkroot!(bratu!, low; precond=lu(A), Δ=1.0, krylov_tol=1e-6, tol=1e-8, maxiter=60, verbose=false)
nkroot!(bratu!, high; precond=lu(A), Δ=1.0, krylov_tol=1e-6, tol=1e-8, maxiter=60, verbose=false)
```

From zero the search finds the lower solution, with a maximum of 0.1405. From a tall guess it finds
the upper one, with a maximum of 4.091. Both solutions are known in closed form, and the computed ones
agree with them to the error of the second differences.

![The two solutions of the Bratu problem](examples/figures/bratu.png)

Script: [`examples/bratu.jl`](examples/bratu.jl).

## The hookstep

The radius Δ bounds the step. On a small radius the step follows the steepest descent, and as Δ grows
it turns towards the Newton step. From the guess `(1.8, 0.7)` the Newton step lands above the root and
raises the residual. The ratio ρ of the actual to the predicted reduction says how far the linear
model can be trusted. The search reads it after each trial.

To see one step on a given radius, fix the radius with `Δmax` and accept every trial with `eta`.

```julia
x = [1.8, 0.7]
nkroot!(two!, x; jacobian=:ad, Δ=0.25, Δmax=0.25, eta=-Inf, maxiter=1, verbose=false)
```

![The step for each radius, and the ratio](examples/figures/hookstep.png)

The dashed circles have radii 0.1, 0.25 and 0.5. The black curve is the end of the step as Δ grows.

Script: [`examples/hookstep.jl`](examples/hookstep.jl).

## The radius

With `trust=:ratio`, the default, the radius moves after each trial. It shrinks by four below
ρ = 1/4 and doubles above ρ = 3/4 on the boundary. With `trust=:scan` the hooksteps at the radii Δ/8
to 4Δ in steps of √2 are all tried in the same Krylov space, and the best is kept. The next radius is
the length of that step.

```julia
x = [2.5, 2.0]
nkroot!(two!, x; jacobian=:ad, trust=:scan, Δ=1.0, verbose=false)
```

A trial costs one residual and no Jacobian product. A scan pays off when the Krylov space costs much
more than a residual, as it does for a long search with many products per step. On these two
equations it only costs more. Three options shape the search.

- `Δ` is the first radius. A small one wastes steps while the radius doubles.
- `Δmax` is the largest radius. A small cap forces a crawl.
- `eta` is the smallest ratio for a step to be accepted. A large one rejects more trials.

![The two rules and the three options](examples/figures/update_rules.png)

Script: [`examples/update_rules.jl`](examples/update_rules.jl).

## The Krylov space

`krylov_tol` sets how well each Newton system is solved. A loose tolerance gives linear convergence
with cheap steps, and a tight one the quadratic convergence of Newton's method at more products per
step. `krylov_dim` caps the size of the space. A space of one vector still converges, about one digit
per step. Here both act on the Bratu problem from zero.

```julia
u = zeros(n)
nkroot!(bratu!, u; precond=lu(A), krylov_tol=1e-2, krylov_dim=10, Δ=1.0, tol=1e-10, maxiter=40, verbose=false)
```

![The effect of krylov_tol and krylov_dim](examples/figures/krylov.png)

Script: [`examples/krylov.jl`](examples/krylov.jl).

## Preconditioners

A right preconditioner is any object with `ldiv!(out, M, v)`. On the Bratu problem the LU factorisation
of the second differences cuts the Jacobian products of a search from 561 to 12, with the same
solution. The left panel shows one Newton system. Without the preconditioner GMRES stagnates until it
has filled half of the space.

Use `flexible=true` with a preconditioner that solves a few directions exactly. It stretches some
vectors far beyond the others, and a Gram matrix of them loses the short ones to rounding. Flexible
GMRES keeps them. The right panel is a linear system with one eigenvalue of $10^{-7}$, and a
preconditioner that is exact on that direction. After twenty steps the Gram matrix is still at $3\times 10^{-7}$.
Flexible GMRES gets to $5\times 10^{-9}$ in one step.

```julia
out = nkroot!(linear!, B, x; precond=M, flexible=true, Δ=1e8, krylov_dim=40, krylov_tol=1e-10)
```

![A preconditioner on the Bratu problem, and flexible GMRES](examples/figures/preconditioner.png)

A preconditioner that depends on the point defines `update!(M, x)`, which the search calls at every
Newton step. The two channels below use one.

Script: [`examples/preconditioner.jl`](examples/preconditioner.jl).

## Tracing

A `Trace` is a callback. After the search it holds the whole history.

```julia
trace = Trace(keep=true)
u     = [20*sin(π*i*h) for i in 1:n]
nkroot!(bratu!, u; precond=lu(A), Δ=20.0, krylov_tol=1e-6, tol=1e-9, maxiter=60, verbose=false, callback=trace)

trace              # prints one row per step
trace.res          # residual norms
trace.iter         # Newton steps
trace.time         # seconds since the start
trace.products     # Jacobian products, cumulative
trace.residuals    # residuals, cumulative
trace.preconditioner # applications of M⁻¹, cumulative
trace.krylov       # the GMRES residual after each Arnoldi step, for each Newton step
trace.step         # "newton" or "hook"
trace.radius       # the radius of each accepted step
trace.steplength   # its length
trace.ratio        # its ratio
trace.rejected     # the rejected trials of each step
trace.trials       # the radius and the ratio of each of them
trace.x            # the iterates, with keep=true
```

Printed, it gives

```
Trace with 27 records
  iter         ‖r‖     radius         ρ  krylov   products   rejected     time
     0  2.4470e+09  2.000e+01       NaN       0          0          0      0.0
     1  9.0019e+08  2.000e+01     0.865      50         50          0      0.0
     2  3.3116e+08  2.000e+01     0.865      50        100          0      0.0
     ⋮
    10  1.6183e+05  2.000e+01     0.712      43        493          0      0.1
    11  1.1560e+05  2.000e+01     0.490      29        522          0      0.1
    12  5.1087e+04  5.000e+00     0.930      26        548          1      0.1
     ⋮
    24  4.3301e-03  2.000e+01     1.000       6        692          0      0.1
    25  6.9034e-07  2.000e+01     1.000       6        698          0      0.1
    26  1.8961e-10  2.000e+01     1.000       6        704          0      0.1
```

and every field can be plotted. Here the search starts far above the solution with a preconditioner
of the Laplacian alone. It walks down on full Krylov spaces, spends a few hooksteps near the fold of
the nonlinearity, and ends with the quadratic phase. The last panel shows the iterates, light to dark.

![Everything a Trace records](examples/figures/tracing.png)

A callback of your own receives the state of the search and stops it by returning `true`. Call the
trace first to keep both. The output can go to a file with `io`, or be silenced with `verbose`.

```julia
trace = Trace()
stop  = info -> (trace(info); info.res < 1e-3 || info.iter >= 15)

u = zeros(n)
open(tempname(), "w") do io
    nkroot!(bratu!, u; precond=lu(A), Δ=1.0, tol=1e-10, maxiter=20, io=io, callback=stop)
end
```

Script: [`examples/tracing.jl`](examples/tracing.jl).

## A turbulent channel

The mean velocity of a turbulent channel from Prandtl's mixing length. In units of the half height
and of the friction velocity, on $0 \le y \le 1$,

$$\frac{d}{dy}\Big[\Big(\frac{1}{Re_\tau} + \ell^2\Big|\frac{dU}{dy}\Big|\Big)\frac{dU}{dy}\Big] = -1,
\qquad U(0) = 0, \qquad \frac{dU}{dy}(1) = 0,$$

with the mixing length of Nikuradse damped at the wall by van Driest,

$$\ell(y) = \big(0.14 - 0.08(1 - y)^2 - 0.06(1 - y)^4\big)\big(1 - e^{-y\,Re_\tau/26}\big).$$

The eddy viscosity depends on the solution, so the problem is nonlinear. Finite volumes on a grid of
400 points stretched towards the wall give a tridiagonal Jacobian. Its LU factorisation, updated at
each step through `update!(M, x)`, is the preconditioner. The search needs about ten Newton steps.
Integrating the equation gives the total stress $1 - y$, a quadratic equation for $dU/dy$ at each
face. The discrete solution is therefore known exactly, and the search reaches it to $10^{-11}$.

![Mean velocity of a channel](examples/figures/channel.png)

Script: [`examples/channel.jl`](examples/channel.jl).

## Spalart–Allmaras

The same channel with a transport model. The streamwise momentum equation and the Spalart–Allmaras
equation for the working variable $\tilde\nu$ are solved together, with $\nu = 1/Re_\tau$,

$$\frac{d}{dy}\Big[(\nu + \nu_t)\frac{dU}{dy}\Big] + 1 = 0,$$

$$c_{b1}\tilde S\tilde\nu - c_{w1}f_w\Big(\frac{\tilde\nu}{y}\Big)^2
+ \frac{1}{\sigma}\Big\{\frac{d}{dy}\Big[(\nu + \tilde\nu)\frac{d\tilde\nu}{dy}\Big]
+ c_{b2}\Big(\frac{d\tilde\nu}{dy}\Big)^2\Big\} = 0,$$

with $\nu_t = \tilde\nu f_{v1}$ and the closures of the model, written out in the script. The unknown
`x` holds both fields.

The stiff term is the diffusion. It is a Laplacian with a variable coefficient, and it is tridiagonal
in each field, so it makes a cheap preconditioner. It depends on the point, so it is factorised again
at every step by `update!`.

```julia
mutable struct Diffusion
    c::Channel # the grid and the viscosity
    A::Any     # LU factorisation of d/dy (ν + νₜ) d/dy
    B::Any     # and of (1/σ) d/dy (ν + ν̃) d/dy
end

function NKRoots.update!(M::Diffusion, x)
    # the two tridiagonal matrices at x, factorised
end

function LinearAlgebra.ldiv!(out, M::Diffusion, v)
    # one solve for each field
end
```

The search goes up in the Reynolds number, and each solution is the guess of the next. Six values
from 180 to 5200 take between 12 and 26 Newton steps each. The solutions satisfy the integrated
balance of the stress to $10^{-11}$, and in the log layer $\tilde\nu^+ = \kappa y^+$, as the model was
built to give. Here is one step of the continuation, from the solution at 550 to the one at 1000.

| preconditioner | Jacobian products | outcome |
|---|---:|---|
| none | 12000 | stops at $10^{-4}$ after 40 steps |
| the Laplacian, $\nu\, d^2/dy^2$, factorised once | 821 | converged |
| the diffusion, factorised at each step | 237 | converged |

The Laplacian alone sees only the viscosity, a hundred times smaller than the eddy viscosity away
from the wall, and the diffusion sees both. A guess matters too. From a parabola at $Re_\tau = 1000$
the search with the diffusion preconditioner stalls, which is why the continuation is there.

![Spalart–Allmaras channel](examples/figures/spalart_allmaras.png)

Script: [`examples/spalart_allmaras.jl`](examples/spalart_allmaras.jl).

## How it works

Newton's method solves $J\,\delta x = -r$ with $r = F(x)$. Far from a root the full step is unreliable,
so the step is restricted to a region where the linear model is trusted,

$$\min_{\delta x} \|r + J\,\delta x\| \quad \text{subject to} \quad \|\delta x\| \le \Delta.$$

The Arnoldi iteration builds an orthonormal basis $Q_n$ of the Krylov space from $-r$, with
$J Q_n = Q_{n+1} H$. For $\delta x = Q_n y$ the problem becomes

$$\min_{y} \|\beta e_1 - H y\| \quad \text{subject to} \quad \|y\| \le \Delta,$$

which the SVD of $H$ solves. The Krylov space stops growing when the GMRES residual falls below
`krylov_tol`, or at `krylov_dim`.

With a right preconditioner the Arnoldi iteration runs on $J M^{-1}$ and the step is
$\delta x = M^{-1} Q_n y$. The trust region still bounds $\delta x$, not $y$. Its metric comes from
the vectors $M^{-1} q_j$, kept by flexible GMRES, or from `gram(M, Q)`, which a preconditioner can
compute cheaply.

The ratio $\rho$ of the actual to the predicted reduction of $\|r\|^2$ judges each trial. A poor trial
gives a shorter hookstep in the same Krylov space, with no new products with $J$.

## Keywords

| keyword | default | meaning | example |
|---|---|---|---|
| `jacobian` | `:fd` | `:fd` or `:ad`, without a Jacobian object | [The Jacobian](#the-jacobian) |
| `ϵ` | `nothing` | relative step of the finite difference, else that of NITSOL | [The Jacobian](#the-jacobian) |
| `precond` | `nothing` | right preconditioner, with `ldiv!` | [Preconditioners](#preconditioners) |
| `flexible` | `false` | keep the vectors `M⁻¹ qⱼ` for the length of the step | [Preconditioners](#preconditioners) |
| `krylov_dim` | `50` | largest dimension of the Krylov space | [The Krylov space](#the-krylov-space) |
| `krylov_tol` | `1e-3` | relative GMRES residual at which the space stops growing | [The Krylov space](#the-krylov-space) |
| `trust` | `:ratio` | `:ratio` or `:scan`, the rule of the radius | [The radius](#the-radius) |
| `Δ` | `1e-2` | first radius of the trust region | [The radius](#the-radius) |
| `Δmax` | `Inf` | largest radius | [The radius](#the-radius) |
| `eta` | `1e-3` | smallest ratio for a step to be accepted | [The radius](#the-radius) |
| `maxiter` | `20` | Newton steps at most | [A first search](#a-first-search) |
| `tol` | `1e-10` | residual norm of convergence | [A first search](#a-first-search) |
| `verbose` | `true` | print one row per step | [Tracing](#tracing) |
| `io` | `stdout` | where to print | [Tracing](#tracing) |
| `callback` | none | called at each step, stops the search by returning `true` | [Tracing](#tracing) |

A search ends with the status `:converged`, `:maxiter`, `:stopped`, or `:collapsed` when no step in the
Krylov space reduces the residual.

## References

- D. Viswanath, *Recurrent motions within plane Couette turbulence*, J. Fluid Mech. 580 (2007).
  [arXiv:physics/0604062](https://arxiv.org/abs/physics/0604062).
- D. Viswanath, *The critical layer in pipe flow at high Reynolds number*, Phil. Trans. R. Soc. A
  367 (2009). [arXiv:0809.1498](https://arxiv.org/abs/0809.1498).
- Y. Saad, *A flexible inner-outer preconditioned GMRES algorithm*, SIAM J. Sci. Comput. 14 (1993).
- P. R. Spalart and S. R. Allmaras, *A one-equation turbulence model for aerodynamic flows*, La
  Recherche Aérospatiale 1 (1994).
