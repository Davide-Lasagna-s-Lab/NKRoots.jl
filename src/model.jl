# The linear model of the Newton system (1) at x, in a Krylov space.
#
# Only the action of J is used. The Arnoldi iteration started from b = -r builds an orthonormal basis
# Qₙ of the Krylov space and the (n+1) × n Hessenberg matrix H with J Qₙ = Qₙ₊₁ H. For δx = Qₙ y,
# ‖r + J δx‖ = ‖g - H y‖ with g = (‖b‖, 0, …, 0) and ‖δx‖ = ‖y‖, so that the trust-region problem (2)
# in the Krylov space is the small problem
#
#     min ‖g - H y‖   subject to   ‖y‖ ≤ Δ,                                          (3)
#
# solved from the SVD of H (hookstep.jl); its solution for Δ = ∞ is the GMRES step. The Krylov space
# stops growing when the residual of the GMRES step falls below `krylov_tol` ‖b‖, monitored by Givens
# rotations of H at no cost, or at dimension `krylov_dim`. The criterion is on the unconstrained
# problem on purpose: on the boundary of the trust region the residual of the hookstep stalls at a
# level set by Δ, which no larger space can lower.
#
# Right preconditioning. With a preconditioner M the Arnoldi iteration runs on J M⁻¹ and the step
# is δx = M⁻¹ Qₙ y: the residual of the model is unchanged, and so is the meaning of `krylov_tol`.
# The trust region still bounds the step δx, not its Krylov coordinates: ‖δx‖ = ‖R y‖ with
# Rᵀ R = G the Gram matrix of the vectors zⱼ = M⁻¹ qⱼ, and (3) keeps its form in u = R y with
# H R⁻¹ in place of H. With `flexible=true` the zⱼ come from the Arnoldi iteration (flexible GMRES,
# Saad 1993, which also allows a preconditioner that changes between Arnoldi steps) and are
# orthonormalised as they come, Zₙ = Wₙ R with Wₙ orthonormal and R upper triangular: the step is
# Wₙ R y, and R keeps the precision of the zⱼ however their lengths spread. Otherwise G comes from
# `gram(M, Qₙ)`, which a preconditioner can compute cheaply, and the step is M⁻¹ (Qₙ y). A Gram
# matrix squares the spread of the lengths, wide when M⁻¹ solves a few slow directions exactly; it
# is scaled to unit diagonal before its root is taken, but some directions are still lost
# (hookstep.jl), and `flexible=true` keeps them. A preconditioner that can give R without forming
# G returns it from `gram`, and loses nothing either.

# The model of (1) at the residual r, in the vectors and the bases of the workspace ws, with the
# counted products by J and applications of M⁻¹. It returns
#   dim            the dimension of the Krylov space,
#   krylov         the relative residual of the GMRES step after each Arnoldi step,
#   hook(Δ)        the coordinates y of the hookstep of radius Δ, and whether it lies on the boundary,
#                  the GMRES step for Δ = ∞,
#   len(y)         the length of the step of coordinates y,
#   move!(out, y)  that step, into out,
#   predicted(y)   the reduction of ‖r‖² that the model predicts for it, β² - ‖g - H y‖².
function _model(r, ws, product!, precondition!; precond, flexible, krylov_dim, krylov_tol)
    # ---- the right-hand side of (1) ----
    ws.b .= .-r
    β     = norm(ws.b)

    # ---- the operator of the Arnoldi iteration, J or J M⁻¹ ----
    empty!(ws.Q)
    empty!(ws.W)
    Rc = Vector{Float64}[]    # flexible: the columns of R, upper triangular
    A! = precond === nothing ? product! :
         flexible            ? (out, v) -> _flexible!(out, product!, precondition!, v, ws.W, Rc, ws.zw) :
                               (out, v) -> product!(out, precondition!(ws.w, v))

    # ---- the Krylov space, one Arnoldi step at a time ----
    arn    = ArnoldiIteration(A!, ws.Q, copy(ws.b))
    g      = [β]
    H      = arn.H
    krylov = Float64[]                       # relative residual of the GMRES step
    qr     = (c=Float64[], s=Float64[], γ=[β]) # Givens QR of H, one column at a time
    for _ in 1:(β > 0 ? krylov_dim : 0)
        # one more basis vector, one more column of H, one more entry of g
        _, H = step!(arn)
        push!(g, 0.0)

        # the GMRES step in the current space: stop once it solves the Newton system well enough
        push!(krylov, _givens!(qr, H[:, end]) / β)
        krylov[end] < krylov_tol && break

        # breakdown: the new direction lies in the space, which no further step can enlarge
        H[end, end] <= 1e-12 * β && break
    end

    # ---- the metric of the step, ‖δx‖ = ‖Rm y‖, without the directions that rounding sets ----
    Hn     = arn.H        # the final H, not captured while the loop grows it
    m      = size(Hn, 2)
    ϵ      = 100 * eps(Float64)
    R      = precond === nothing || !flexible ? nothing :
             UpperTriangular([i <= j ? Rc[j][i] : 0.0 for i in 1:m, j in 1:m])
    Rm, Ri = precond === nothing ? (I, I) :
             flexible            ? _root(R, ϵ) : _root(_metric(gram(precond, view(ws.Q, 1:m))), ϵ)

    # ---- the hookstep (3) for any radius, from the SVD of H in the metric of the step ----
    SH = svd(Hn * Ri)
    hook(Δ)       = (h = _hookstep(SH, g, Δ); (Ri * h[1], h[2]))
    len(y)        = norm(Rm * y)
    move!(out, y) = precond === nothing ? lincomb!(out, ws.Q, y) :
                    flexible            ? lincomb!(out, ws.W, R * y) :
                                          precondition!(out, lincomb!(ws.z, ws.Q, y))
    predicted(y)  = β^2 - norm(g - Hn * y)^2

    return (dim       = m,
            krylov    = krylov,
            hook      = hook,
            len       = len,
            move!     = move!,
            predicted = predicted)
end
