module LTVModels

using DSP

using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile

abstract Model

type LTVModel <: Model
    At
    Bt
end


include("utilities.jl")
include("peakdetection.jl")
include("fit.jl")
include("seg_bellman.jl")
include("kalmanmodel.jl")
include("gmmmodel.jl")

end # module
