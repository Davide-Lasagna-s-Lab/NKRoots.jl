# Trace: a record of a search, passed as the callback of nkroot!, as the Trace of ReSolver.jl.

export Trace

"""
    Trace(; keep=false)

History of a search, filled when passed as `callback` to [`nkroot!`](@ref). At the start and
after every iteration it records

- `iter`, the iteration number, and `time`, the seconds elapsed since the first record;
- `res`, the residual norm `‖r‖`;
- `residuals`, `products` and `preconditioner`, the cumulative numbers of evaluations of the
  residual by the search, of products with the Jacobian and of applications of the preconditioner;
- `jacobian_updates`, `preconditioner_updates` and `grams`, the cumulative numbers of moves of the
  Jacobian, `update!(J, x, r)`, and of the preconditioner, `update!(M, x)`, and of calls of `gram`;
- `function_calls`, the cumulative number of calls of `F`, those of the search and those that a
  Jacobian built by NKRoots.jl makes, by finite differences or dual numbers; the calls inside a
  Jacobian of your own are not seen;
- `krylov`, the relative residual of the GMRES solution of the Newton system after each Arnoldi
  step of the iteration (empty at the start);
- `step`, the kind of the accepted step: `"newton"` if it is the GMRES step, `"hook"` if it lies on
  the boundary of the trust region, `"line"` if backtracking has shortened it (empty at the start);
- `steplength`, the length `‖δx‖` of the accepted step, in the `norm` of the vectors, also when a
  preconditioner is used (`NaN` at the start);
- `fraction`, the fraction of the GMRES step that the accepted step is, its length over the length
  of the GMRES step of the iteration, both in the `norm` of the vectors. It is 1 for a `"newton"`
  step. With backtracking it is the λ by which the GMRES step was shortened, the cap of `Δmax`
  included. For a `"hook"` step it is below 1, unless the geodesic acceleration, which is part of
  the step and of its length, makes it longer (`NaN` at the start);
- `radius`, the trust-region radius with which the accepted step was computed. A `"newton"` step
  is shorter than its radius, a `"hook"` step is as long. Without a trust region, with
  `globalisation=:backtracking` or `:none`, it is the length of the step again. At the start it
  is the initial radius `Δ`;
- `globalisation`, the keyword `globalisation` of the search that made the record;
- `ratio`, the ratio of the actual to the predicted reduction of `‖r‖²` of the accepted step
  (`NaN` at the start);
- `trials`, the other trials of the iteration, in the order they were made, each as the pair of
  its radius and its ratio. With backtracking these are the steps that were refused before the
  accepted one, each half as long as the one before and with its length as radius. The accepted
  step is then the fraction `2^-length(trials)` of the GMRES step, which is `fraction`, unless
  `Δmax` has capped it first;
- `rejected`, the number of those trials with a ratio not above `eta`. It is `length(trials)` with
  backtracking and with `trust=:classical`. With `trust=:scan` some trials are good steps that
  were not chosen, and it is smaller;
- `x`, a copy of the iterate, only with `keep=true`.

Printed, a trace is a table with one row per record, with the columns `‖F‖`, `‖δx‖` for
`steplength`, `Δ` for `radius`, `λ` for `fraction`, `ρ` for `ratio`, the dimension of the Krylov
space, the cumulative products, the rejected trials and the time. The `Δ` cell is blank in the
records of a search without a trust region, where the field `radius` holds the length of the step.

The same trace can follow several calls to `nkroot!`; the iteration numbers restart at zero with
each call. A callback that also stops the search can call the trace first:
`callback = info -> (trace(info); stop(info))`.

```julia
trace = Trace()
nkroot!(F!, J, x; callback=trace)
trace.res      # residual norms, in order
trace.products # Jacobian products, cumulative
```
"""
struct Trace
                      iter::Vector{Int}                             # iteration number
                      time::Vector{Float64}                         # seconds since the first record
                       res::Vector{Float64}                         # residual norm ‖r‖
                 residuals::Vector{Int}                             # residual evaluations, cumulative
                  products::Vector{Int}                             # Jacobian products, cumulative
            preconditioner::Vector{Int}                             # preconditioner applications, cumulative
          jacobian_updates::Vector{Int}                             # moves of J, cumulative
    preconditioner_updates::Vector{Int}                             # moves of M, cumulative
                     grams::Vector{Int}                             # calls of gram, cumulative
            function_calls::Vector{Int}                             # calls of F, cumulative
                    krylov::Vector{Vector{Float64}}                 # relative GMRES residuals of the iteration
                      step::Vector{String}                          # "newton", "hook" or "line"
                    radius::Vector{Float64}                         # radius of the accepted step
                steplength::Vector{Float64}                         # length of the accepted step
                  fraction::Vector{Float64}                         # its length over that of the GMRES step
                     ratio::Vector{Float64}                         # actual over predicted reduction of ‖r‖²
                  rejected::Vector{Int}                             # trials with a ratio below eta
                    trials::Vector{Vector{Tuple{Float64, Float64}}} # radius and ratio of the other trials
             globalisation::Vector{Symbol}                          # globalisation of the search
                         x::Vector{Any}                             # copies of the iterates, with keep=true
                      keep::Bool                                    # keep the iterates
                        t₀::Base.RefValue{Float64}                  # wall-clock time of the first record

    Trace(; keep::Bool=false) = new(Int[],
                                    Float64[],
                                    Float64[],
                                    Int[],
                                    Int[],
                                    Int[],
                                    Int[],
                                    Int[],
                                    Int[],
                                    Int[],
                                    Vector{Float64}[],
                                    String[],
                                    Float64[],
                                    Float64[],
                                    Float64[],
                                    Float64[],
                                    Int[],
                                    Vector{Tuple{Float64, Float64}}[],
                                    Symbol[],
                                    Any[],
                                    keep,
                                    Ref(0.0))
end

# record one iteration; never stops the search
function (trace::Trace)(info::NamedTuple)
    isempty(trace.time) && (trace.t₀[] = time())

    push!(trace.iter,           info.iter)
    push!(trace.time,           time() - trace.t₀[])
    push!(trace.res,            info.res)
    push!(trace.residuals,      info.evaluations.residuals)
    push!(trace.products,       info.evaluations.products)
    push!(trace.preconditioner, info.evaluations.preconditioner)
    push!(trace.jacobian_updates,       info.evaluations.jacobian_updates)
    push!(trace.preconditioner_updates, info.evaluations.preconditioner_updates)
    push!(trace.grams,                  info.evaluations.grams)
    push!(trace.function_calls,         info.evaluations.function_calls)
    push!(trace.krylov,         copy(info.krylov))
    push!(trace.step,           info.step)
    push!(trace.radius,         info.radius)
    push!(trace.steplength,     info.steplength)
    push!(trace.fraction,       info.fraction)
    push!(trace.ratio,          info.ratio)
    push!(trace.rejected,       info.rejected)
    push!(trace.trials,         copy(info.trials))
    push!(trace.globalisation,  info.globalisation)
    trace.keep && push!(trace.x, copy(info.x))

    return false
end

Base.length(trace::Trace) = length(trace.res)

# a table, one row per record
function Base.show(io::IO, ::MIME"text/plain", trace::Trace)
    println(io, "Trace with $(length(trace)) records")
    length(trace) == 0 && return

    names  = ("iter", "‖F‖", "‖δx‖", "Δ", "λ", "ρ", "dim K", "products", "rej.", "time")
    widths = (4, 10, 9, 9, 9, 7, 5, 8, 4, 7)

    println(io, _rule("╭", "┬", "╮", widths))
    println(io, _cells(names, widths))
    println(io, _rule("├", "┼", "┤", widths))
    for i in eachindex(trace.res)
        first  = trace.iter[i] == 0
        region = trace.globalisation[i] === :hookstep # the radius is one of a trust region
        cells  = (string(trace.iter[i]),
                  @sprintf("%.4e", trace.res[i]),
                  first ? "" : @sprintf("%.2e", trace.steplength[i]),
                  region ? @sprintf("%.2e", trace.radius[i]) : "",
                  first ? "" : @sprintf("%.2e", trace.fraction[i]),
                  first ? "" : @sprintf("%.3f", trace.ratio[i]),
                  first ? "" : string(length(trace.krylov[i])),
                  string(trace.products[i]),
                  first ? "" : string(trace.rejected[i]),
                  @sprintf("%.1f", trace.time[i]))
        println(io, _cells(cells, widths))
    end
    print(io, _rule("╰", "┴", "╯", widths))
end
