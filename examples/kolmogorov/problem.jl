# Equilibria of Kolmogorov flow: the discretisation of notes/kolmogorov/kolmogorov.pdf. The script
# kolmogorov.jl and the tests of the package include this file.
#
# The flow is two-dimensional and periodic on the square [0, 2π)², driven by the force sin(4y) eₓ
# (Golden 2026, section 2). With the vorticity ω = ∂ₓv - ∂ᵧu, the streamfunction ∇²ψ = -ω and the
# velocity u = ∂ᵧψ, v = -∂ₓψ, an equilibrium is a root of
#
#     N(ω) = u·∇ω - (1/Re) ∇²ω + 4 cos(4y).
#
# The search runs in the subspace of the fields invariant under the rotation (x, y) → (-x, -y),
# where a state cannot drift along x. The unknown is the vorticity on the grid, and the residual is
#
#     G(ω) = Π N(Πω) + (I - Π) ω,
#
# with Π the projector that discards the modes beyond the two-thirds rule, removes the mean and takes
# the real part of every Fourier coefficient, which is the symmetric part of the field.
using FFTW
using LinearAlgebra
using Random
using NKRoots


# ---- the grid, the wavenumbers and the transforms ----
struct Kolmogorov{P, Q}
         Re::Float64         # Reynolds number
          N::Int             # points along each side
         kx::Matrix{Float64} # wavenumber along x of each coefficient of the real transform
         ky::Matrix{Float64} # wavenumber along y
         k²::Matrix{Float64} # kx² + ky², with 1 at the mean so that it can divide
       keep::Matrix{Float64} # 1 on the kept modes, 0 on the discarded ones and on the mean
    forcing::Matrix{Float64} # 4 cos(4y) on the grid
       wave::Matrix{Float64} # sin(4y) on the grid, the force along x
       plan::P               # real transform of a field on the grid
      iplan::Q               # and its inverse

    function Kolmogorov(N::Int=128; Re::Real=40.0)
        # ---- the wavenumbers: 0 to N/2 along x, where the real transform keeps half, all along y ----
        kx = [Float64(i) for i in 0:N÷2, j in 1:N]
        ky = [Float64(j <= N÷2 ? j : j - N) for i in 0:N÷2, j in 0:N - 1]
        k² = kx.^2 .+ ky.^2

        # ---- the two-thirds rule on each wavenumber, and no mean ----
        K       = N ÷ 3
        keep    = Float64.((abs.(kx) .<= K) .& (abs.(ky) .<= K) .& (k² .> 0))
        k²[1, 1] = 1.0

        # ---- the force, and the forcing of the vorticity ----
        y       = [2π*j/N for i in 1:N, j in 0:N - 1]
        forcing = 4 .* cos.(4 .* y)
        wave    = sin.(4 .* y)

        plan  = plan_rfft(zeros(N, N))
        iplan = plan_irfft(zeros(ComplexF64, N÷2 + 1, N), N)

        return new{typeof(plan), typeof(iplan)}(Float64(Re),
                                                N,
                                                kx,
                                                ky,
                                                k²,
                                                keep,
                                                forcing,
                                                wave,
                                                plan,
                                                iplan)
    end
end

# the coefficients of Πω: kept modes, no mean, real parts
project(K::Kolmogorov, ω) = K.keep .* real.(K.plan * ω)

# the field of the coefficients a
field(K::Kolmogorov, a) = K.iplan * complex.(a)

# the two components of the velocity of the coefficients a of the vorticity, on the grid
function velocity(K::Kolmogorov, a)
    ψ = a ./ K.k²
    u = K.iplan * (im .* K.ky .* ψ)
    v = K.iplan * (-im .* K.kx .* ψ)

    return u, v
end


# ---- the residual G(ω), written into out ----
function residual!(out, ω, K::Kolmogorov)
    # ---- Πω, its velocity and its gradient ----
    a    = project(K, ω)
    u, v = velocity(K, a)
    ωx   = K.iplan * (im .* K.kx .* a)
    ωy   = K.iplan * (im .* K.ky .* a)

    # ---- the advection and the forcing, on the grid, and their coefficients in the range of Π ----
    n = project(K, u .* ωx .+ v .* ωy .+ K.forcing)

    # ---- Π N(Πω) - Πω, with the viscous term, and ω: the identity outside the range of Π ----
    out .= ω .+ field(K, n .+ (K.k² ./ K.Re .- 1) .* a)

    return out
end


# ---- the preconditioner: the Stokes operator on the range of Π, the identity elsewhere ----
struct Stokes{T}
    K::T # the grid
end

# M⁻¹ v: the coefficients in the range of Π are multiplied by Re/|k|²
function LinearAlgebra.ldiv!(out, M::Stokes, v)
    K    = M.K
    a    = project(K, v)
    out .= v .+ field(K, (K.Re ./ K.k² .- 1) .* a)

    return out
end


# ---- the laminar state, ω = -(Re/4) cos(4y) ----
laminar(K::Kolmogorov) = -(K.Re/16) .* K.forcing


# ---- a guess: noise filtered at max(|kx|, |ky|) ≤ cutoff, symmetric, of zero mean and unit deviation ----
function guess(K::Kolmogorov, seed::Int; cutoff::Int=(4, 5, 6)[mod1(seed, 3)])
    noise = randn(Xoshiro(seed), K.N, K.N)
    low   = (abs.(K.kx) .<= cutoff) .& (abs.(K.ky) .<= cutoff)
    ω     = field(K, project(K, noise) .* low)

    return ω ./ sqrt(sum(abs2, ω) / length(ω))
end


# ---- the energy, the dissipation ⟨ω²⟩/Re and the input ⟨u sin(4y)⟩ of the field ω ----
function budget(K::Kolmogorov, ω)
    a    = project(K, ω)
    u, v = velocity(K, a)
    mean(f) = sum(f) / length(f)

    return (energy      = mean(u.^2 .+ v.^2) / 2,
            dissipation = mean(field(K, a).^2) / K.Re,
            input       = mean(u .* K.wave))
end
