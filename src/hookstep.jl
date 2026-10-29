# The trust-region problem in the Krylov space, as in ReSolver.jl.
#
# With the SVD H = U D Vᵀ and p = Uᵀg, min ‖g - H y‖ subject to ‖y‖ ≤ Δ decouples along the
# singular directions. Its solution is
#
#     y(μ) = V (p d / (d² + μ)),     ‖y(μ)‖ = ‖p d / (d² + μ)‖,
#
# where μ ≥ 0 is the Lagrange multiplier of the constraint: μ = 0 gives the least-squares (GMRES)
# solution V (p / d), and larger μ damp the directions of small singular values, shortening the
# step and turning it towards steepest descent. If the least-squares solution lies inside the trust
# region it is the answer; otherwise μ is the root of ‖y(μ)‖ = Δ, which decreases monotonically in
# μ and is found by bisection on [0, ‖p‖ max(d) / Δ], an interval where ‖y(μ)‖ ≤ ‖p‖ max(d) / μ
# guarantees a sign change. Returns y and whether it lies on the boundary. The SVD of H is given,
# to be reused for several radii in the same space.
function _hookstep(SH::SVD, g::Vector, Δ::Real)
    U, d, V = SH
    p       = U' * g

    # ---- ‖y(μ)‖ - Δ, decreasing in μ ----
    fun(μ) = norm(p .* d ./ (d.^2 .+ μ)) - Δ

    # ---- least-squares solution, if inside the trust region ----
    fun(0.0) <= 0 && return V * (p ./ d), false

    # ---- boundary: root of fun by bisection ----
    lo = 0.0
    hi = norm(p) * maximum(d) / Δ
    for _ in 1:100
        μ = (lo + hi) / 2
        fun(μ) > 0 ? (lo = μ) : (hi = μ)
    end

    # hi keeps ‖y‖ ≤ Δ
    return V * (p .* d ./ (d.^2 .+ hi)), true
end

# R and R⁻¹ for the length of a step, ‖Zₙ y‖ = ‖R y‖ with Zₙ the vectors zⱼ = M⁻¹ qⱼ, without the
# directions that rounding sets: there the zⱼ depend on each other, the hookstep would take long
# steps for nothing, and R⁻¹ maps the coordinates u of the kept directions back to y. The columns
# are scaled to unit length first, D the lengths of the zⱼ, so that the short vectors are kept
# beside the long ones that a preconditioner stretches when it solves a few slow directions
# exactly; a cut relative to the largest length would drop them, and with them the GMRES step.
#
# From R, upper triangular with Zₙ = Wₙ R (flexible): the SVD of R D⁻¹, without the singular values
# below ϵ times the largest.
function _root(R::UpperTriangular, ϵ::Real)
    # ---- the lengths of the columns of R, those of the vectors zⱼ ----
    D = [max(norm(view(R, :, j)), floatmin()) for j in axes(R, 2)]

    # ---- the SVD of R D⁻¹, without the directions that rounding sets ----
    _, s, V = svd(R ./ D')
    keep    = s .> ϵ * maximum(s)
    s       = s[keep]
    V       = V[:, keep]

    # ‖R y‖ = ‖diag(s) Vᵀ D y‖
    return s .* V' .* D', (V ./ s') ./ D
end

# From the Gram matrix G = Rᵀ R (gram): the eigenvalues of D⁻¹ G D⁻¹, without those below ϵ times
# the largest. A Gram matrix squares the condition of the vectors, so the directions along which
# the zⱼ, scaled to unit length, depend on each other below √ϵ are lost; flexible GMRES keeps them.
function _root(G::Symmetric, ϵ::Real)
    # ---- the lengths of the vectors zⱼ ----
    D = [sqrt(max(G[j, j], floatmin())) for j in axes(G, 2)]

    # ---- the eigenvalues of D⁻¹ G D⁻¹, without the directions that rounding sets ----
    λ, V = eigen(Symmetric(G ./ D ./ D'))
    keep = λ .> ϵ * maximum(λ)
    s    = sqrt.(λ[keep])
    V    = V[:, keep]

    # ‖R y‖ = ‖diag(s) Vᵀ D y‖
    return s .* V' .* D', (V ./ s') ./ D
end
