module LTVModels
using LTVModelsBase
import LTVModelsBase: AbstractModel, AbstractCost, ModelAndCost, f,
dc,calculate_cost,calculate_final_cost,
fit_model, predict, df,costfun, LTVStateSpaceModel,
SimpleLTVModel

export predict, simulate, fit_model

export KalmanModel, GMMModel


using DSP, Plots, Juno#, Convex, FirstOrderSolvers
using Base.Test

using ReverseDiff: GradientTape, GradientConfig, gradient!

"""
Struct representing a statespace model fit using a Kalman smoother
# Fields
- `At, Bt, Pt` all of type `Array{T,3}` with time in the last dimension
- `extended::Bool` indicates whether or not the model has been extended by one time step to match the length of the input data.
"""
mutable struct KalmanModel{T} <: LTVStateSpaceModel
    At::Array{T,3}
    Bt::Array{T,3}
    Pt::Array{T,3}
    extended::Bool
end
function KalmanModel{T}(At::Array{T,3},Bt::Array{T,3},Pt::Array{T,3},extend::Bool)
    if extend
        At = cat(3,At,At[:,:,end])
        Bt = cat(3,Bt,Bt[:,:,end])
        Pt = cat(3,Pt,Pt[:,:,end])
    end
    return KalmanModel{T}(At,Bt,Pt,extend)
end

KalmanModel(At,Bt,Pt,extend::Bool=false) = KalmanModel{eltype(At)}(At,Bt,Pt,extend)

mutable struct GMMModel <: AbstractModel
    M
    dynamics
    T
end

include("utilities.jl")
include("arx.jl")
include("peakdetection.jl")
include("statespace_fit.jl")
include("seg_bellman.jl")
include("kalmanmodel.jl")
# include("gmmmodel.jl") # Not yet working with julia v0.6
include("statespace_utils.jl")
include("wrappers.jl")
end # module
