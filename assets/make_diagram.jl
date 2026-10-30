# Draws assets/newton_step.svg: one Newton step of the solver, with the colour of each box saying what
# it costs. The Krylov space is paid for once, and the trials inside it cost one residual each.
#
#     julia assets/make_diagram.jl

using Printf


# ---- the colours say the cost ----

const INK    = "#242936"
const GREY   = "#98a2b3"
const BLUE   = "#4063d8"
const GREEN  = "#389826"
const PURPLE = "#9558b2"
const RED    = "#cb3c33"
const FONT   = "font-family=\"Inter, Avenir, Helvetica, Arial, sans-serif\""


# ---- a box with a title and a line below it, a label, an arrow ----

function box(x, y, w, h, title, line, colour)
    return @sprintf("<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" rx=\"14\" fill=\"#ffffff\" stroke=\"%s\" stroke-width=\"3.5\"/>", x, y, w, h, colour) *
           @sprintf("<text x=\"%d\" y=\"%d\" text-anchor=\"middle\" %s font-size=\"20\" font-weight=\"760\" fill=\"#222734\">%s</text>", x + w/2, y + h/2 - 6, FONT, title) *
           @sprintf("<text x=\"%d\" y=\"%d\" text-anchor=\"middle\" %s font-size=\"16\" font-weight=\"560\" fill=\"#4f596b\">%s</text>", x + w/2, y + h/2 + 20, FONT, line)
end

label(x, y, text; anchor="start", size=16, colour="#4f596b") =
    @sprintf("<text x=\"%d\" y=\"%d\" text-anchor=\"%s\" %s font-size=\"%d\" font-weight=\"600\" fill=\"%s\">%s</text>", x, y, anchor, FONT, size, colour, text)

arrow(path; colour=INK) =
    @sprintf("<path d=\"%s\" fill=\"none\" stroke=\"%s\" stroke-width=\"3.2\" stroke-linejoin=\"round\" marker-end=\"url(#head)\"/>", path, colour)

legend(x, y, colour, text) =
    @sprintf("<rect x=\"%d\" y=\"%d\" width=\"22\" height=\"14\" rx=\"4\" fill=\"#ffffff\" stroke=\"%s\" stroke-width=\"3\"/>", x, y - 12, colour) *
    label(x + 30, y, text; size=15)


# ---- the boxes ----

boxes = [box(40, 70, 190, 88, "the point", "x, r = F(x)", GREY),
         box(270, 70, 190, 88, "update!", "J and M move to x", GREY),
         box(500, 70, 190, 88, "Arnoldi on J M⁻¹", "one product a vector", BLUE),
         box(730, 70, 190, 88, "SVD of H", "H = U D Vᵀ", PURPLE),
         box(730, 240, 190, 88, "hookstep", "μ from ‖s(μ)‖ = Δ", PURPLE),
         box(500, 240, 190, 88, "trial", "F(x + δ) and ρ", GREEN)]

decision = "<path d=\"M365 222 L445 284 L365 346 L285 284 Z\" fill=\"#ffffff\" stroke=\"$RED\" stroke-width=\"3.5\" stroke-linejoin=\"round\"/>" *
           label(365, 291, "ρ &gt; eta ?"; anchor="middle", size=20, colour="#222734")


# ---- the arrows ----

arrows = [arrow("M230 114 L266 114"),
          arrow("M460 114 L496 114"),
          arrow("M690 114 L726 114"),
          arrow("M825 158 L825 236"),
          arrow("M730 284 L694 284"),
          arrow("M500 284 L449 284"),
          arrow("M285 284 L135 284 L135 162"; colour=GREEN),
          arrow("M365 346 L365 382 L825 382 L825 332"; colour=RED)]

labels = [label(135, 52, "✓ stop when ‖r‖ &lt; tol"; anchor="middle"),
          label(150, 272, "yes: accept x + δ,"; colour=GREEN),
          label(150, 306, "update Δ"; colour=GREEN),
          label(387, 372, "no: Δ becomes Δ/4, a new hookstep, no product with J"; colour=RED)]

keys = [legend(40, 392, BLUE, "products with J"),
        legend(40, 370, GREEN, "one residual"),
        legend(40, 348, PURPLE, "small dense algebra")]


# ---- the svg ----

svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 420" role="img" aria-labelledby="title desc">
  <title id="title">One Newton step of NKRoots.jl</title>
  <desc id="desc">From the point x the Jacobian and the preconditioner are updated, the Arnoldi iteration builds a Krylov space, and the singular value decomposition of its small matrix gives hooksteps. Each hookstep is tried with one residual. A step accepted returns to the top, a step rejected shrinks the radius and returns to the hookstep with no new product.</desc>
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#ffffff"/>
      <stop offset="1" stop-color="#f2f5f8"/>
    </linearGradient>
    <marker id="head" viewBox="0 0 12 10" refX="9" refY="5" markerWidth="4.2" markerHeight="3.6" orient="auto-start-reverse">
      <path d="M0 0 L12 5 L0 10 Z" fill="context-stroke"/>
    </marker>
  </defs>

  <rect x="8" y="8" width="944" height="404" rx="22" fill="url(#bg)" stroke="#e4e8ee" stroke-width="1.5"/>
  $(join(arrows, "\n  "))
  $(join(boxes, "\n  "))
  $decision
  $(join(labels, "\n  "))
  $(join(keys, "\n  "))
</svg>
"""

write(joinpath(@__DIR__, "newton_step.svg"), svg)
println("written assets/newton_step.svg")
