module NKRoots

using LinearAlgebra
using Printf

include("krylov.jl")
include("preconditioner.jl")
include("hookstep.jl")
include("model.jl")
include("globalisation.jl")
include("output.jl")
include("solver.jl")
include("jacobian.jl")
include("trace.jl")

end
