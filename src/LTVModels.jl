module LTVModels


export predict, simulate, fit_model

using DSP, Plots, Convex, FirstOrderSolvers

using ReverseDiff: GradientTape, GradientConfig,GradientResult, gradient!
import DiffBase



include("interfaces.jl")
include("utilities.jl")
include("peakdetection.jl")
include("statespace_fit.jl")
include("seg_bellman.jl")
include("kalmanmodel.jl")
# include("gmmmodel.jl") # Not yet working with julia v0.6
include("statespace_utils.jl")
include("wrappers.jl")
end # module
