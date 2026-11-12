# Draws assets/logo.svg: a landscape of ‖F‖ around a root, and a search that walks down to it inside
# trust regions that shrink as the linear model improves, each step a hookstep.
#
#     julia assets/make_logo.jl

using Printf


# ---- the landscape, ellipses around the root ----

root   = (74.0, 20.0)    # where the search ends
levels = [(22.0, 11.0), (48.0, 25.0), (78.0, 42.0), (110.0, 60.0), (144.0, 80.0)]
tilt   = -24.0           # angle of the valley

contours = [@sprintf("<ellipse cx=\"0\" cy=\"0\" rx=\"%.1f\" ry=\"%.1f\" fill=\"none\" stroke=\"#c6ccd6\" stroke-width=\"%.1f\"/>",
                     a, b, 2.6 - 0.25*i) for (i, (a, b)) in enumerate(levels)]


# ---- the search: each iterate with its trust region, the first with the largest ----

COLORS = ["#389826", "#4063d8", "#9558b2", "#cb3c33"]
points = [(-128.0, -52.0), (-66.0, 40.0), (-6.0, 58.0), (44.0, 40.0)]
radii  = [50.0, 37.0, 27.0, 19.0]

regions = [@sprintf("<circle cx=\"%.1f\" cy=\"%.1f\" r=\"%.1f\" fill=\"%s\" fill-opacity=\"0.14\" stroke=\"%s\" stroke-width=\"4\" stroke-dasharray=\"9 7\"/>",
                    p[1], p[2], r, c, c) for (p, r, c) in zip(points, radii, COLORS)]

# a step from one iterate to the next, bent as a hookstep is
function hook(a, b)
    mx = (a[1] + b[1])/2 - 0.22*(b[2] - a[2])
    my = (a[2] + b[2])/2 + 0.22*(b[1] - a[1])
    return @sprintf("M%.1f %.1f Q%.1f %.1f %.1f %.1f", a[1], a[2], mx, my, b[1], b[2])
end

route = [points; root]
steps = [@sprintf("<path d=\"%s\" fill=\"none\" stroke=\"#242936\" stroke-width=\"5\" stroke-linecap=\"round\" marker-end=\"url(#flow-arrow)\"/>",
                  hook(route[i], route[i + 1])) for i in 1:length(points)]
dots  = [@sprintf("<circle cx=\"%.1f\" cy=\"%.1f\" r=\"6\" fill=\"#242936\"/>", p[1], p[2]) for p in points]


# ---- the root, a five-pointed star ----

star = join([@sprintf("%.1f %.1f", root[1] + (isodd(k) ? 8.0 : 19.0)*sin(k*π/5), root[2] - (isodd(k) ? 8.0 : 19.0)*cos(k*π/5))
              for k in 0:9], " L")


# ---- the svg ----

svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 300" role="img" aria-labelledby="title desc">
  <title id="title">NKRoots.jl logo</title>
  <desc id="desc">A search walks down the contours of a residual to a root, a star, each step a hookstep inside a dashed trust region in Julia colors, beside the NKRoots.jl wordmark.</desc>
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#ffffff"/>
      <stop offset="1" stop-color="#f2f5f8"/>
    </linearGradient>
    <filter id="panel-shadow" x="-12%" y="-25%" width="124%" height="150%">
      <feDropShadow dx="0" dy="9" stdDeviation="11" flood-color="#172033" flood-opacity="0.14"/>
    </filter>
    <filter id="soft" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#172033" flood-opacity="0.10"/>
    </filter>
    <marker id="flow-arrow" viewBox="0 0 12 10" refX="9" refY="5" markerWidth="4.0" markerHeight="3.4" orient="auto">
      <path d="M0 0 L12 5 L0 10 Z" fill="#242936"/>
    </marker>
  </defs>

  <!-- Main panel -->
  <rect x="24" y="24" width="912" height="252" rx="24" fill="url(#bg)" filter="url(#panel-shadow)"/>

  <!-- Left illustration: a search inside its trust regions, on the landscape of the residual -->
  <g transform="translate(232 150)">
    <g transform="translate($(root[1]) $(root[2])) rotate($tilt)">
      $(join(contours, "\n      "))
    </g>
    <g filter="url(#soft)">
      $(join(regions, "\n      "))
      $(join(steps, "\n      "))
      $(join(dots, "\n      "))
      <path d="M$star Z" fill="#242936"/>
    </g>
  </g>

  <!-- Wordmark and subtitle -->
  <g transform="translate(476 96)">
    <text x="0" y="48" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="58" font-weight="820" fill="#222734" letter-spacing="0">NKRoots.jl</text>
    <text x="2" y="90" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="21" font-weight="700" fill="#4f596b" letter-spacing="0">Newton–Krylov solver for F(x) = 0,</text>
    <text x="2" y="118" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="21" font-weight="700" fill="#4f596b" letter-spacing="0">globalised, preconditioned and traced.</text>
  </g>
</svg>
"""

write(joinpath(@__DIR__, "logo.svg"), svg)
println("written assets/logo.svg")
