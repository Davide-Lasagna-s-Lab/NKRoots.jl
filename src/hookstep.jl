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

# R and R⁻¹ with Rᵀ R = G, the Gram matrix of the steps of the Krylov coordinates, from its
# eigenvalues, without the directions of eigenvalues below ϵ times the largest, which rounding
# sets: there the steps depend on each other, and R⁻¹ maps the coordinates u of the kept
# directions back to y. Then ‖R y‖ is the length of the step of y.
function _root(G::AbstractMatrix, ϵ::Real)
    λ, V = eigen(Symmetric(G))
    keep = λ .> ϵ * maximum(λ)
    s    = sqrt.(λ[keep])
    return s .* V[:, keep]', V[:, keep] ./ s'
end
