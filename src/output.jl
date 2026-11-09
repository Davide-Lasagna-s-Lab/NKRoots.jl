# The table that nkroot! prints, one row per Newton step. ‖F‖: residual norm. Δ: trust-region
# radius after the update, or λ, the fraction of the GMRES step taken, without a trust region. ρ:
# ratio of the actual to the predicted reduction. dim K: dimension of the Krylov space. GMRES res.:
# relative residual of the GMRES solution of the Newton system, the full step before the
# globalisation. step: ↪ hook on the boundary of the trust region, → newton the GMRES step, ↓ line
# the GMRES step shortened by backtracking. rej.: trial steps rejected before the accepted one. The
# last line says how the search ended, ✗ when no step reduces ‖F‖.

const _ROWS  = ("iter", "‖F‖", "Δ", "ρ", "dim K", "GMRES res.", "step", "rej.")
const _WIDTH = (4, 10, 8, 7, 5, 10, 8, 4)
const _KINDS = Dict("hook" => "↪ hook", "newton" => "→ newton", "line" => "↓ line", "" => "")

# a rule of the table, from its left, middle and right corners
_rule(l, m, r, widths) = l * join(("─"^(w + 2) for w in widths), m) * r

# a row of the table, each cell right-aligned
_cells(cells, widths) = "│ " * join((lpad(c, w) for (c, w) in zip(cells, widths)), " │ ") * " │"

# the top of the table, with Δ or λ in the third column
function _print_header(io::IO, globalisation)
    names = globalisation === :hookstep ? _ROWS : Base.setindex(_ROWS, "λ", 3)

    println(io, _rule("╭", "┬", "╮", _WIDTH))
    println(io, _cells(names, _WIDTH))
    println(io, _rule("├", "┼", "┤", _WIDTH))
end

# one row; the first, at the start, holds the residual and the radius only
function _print_row(io::IO, iter, res, Δ, ρ, k, lin, step, rejected)
    first = iter == 0
    cells = (string(iter),
             @sprintf("%.4e", res),
             @sprintf("%.2e", Δ),
             first ? "" : @sprintf("%.3f", ρ),
             first ? "" : string(k),
             first ? "" : @sprintf("%.2e", lin),
             _KINDS[step],
             first ? "" : string(rejected))
    println(io, _cells(cells, _WIDTH))
    flush(io)
end

# the bottom of the table, and how the search ended
function _print_footer(io::IO, status, iter, res, counts, globalisation)
    collapsed = globalisation === :hookstep ? "✗ trust region collapsed" : "✗ no step accepted"
    mark      = status === :converged ? "✓ converged" :
                status === :collapsed ? collapsed :
                status === :maxiter   ? "✗ maxiter reached" : "■ stopped by the callback"
    println(io, _rule("╰", "┴", "╯", _WIDTH))
    @printf(io, "  %s: %d steps, ‖F‖ = %.3e, %d residuals, %d Jacobian products\n",
            mark, iter, res, counts.residuals, counts.products)
    flush(io)
end
