module LTVModels
using LinearAlgebra, Statistics, Printf, Random, Distributions
using LinearTimeVaryingModelsBase, ControlSystemIdentification
import LinearTimeVaryingModelsBase: AbstractModel, AbstractCost, ModelAndCost, f,
dc,calculate_cost,calculate_final_cost,
predict, simulate, df,costfun, LTVStateSpaceModel,
SimpleLTVModel, rms, sse, nrmse, modelfit, aic

export predict, simulate

export KalmanModel, KalmanAR, rootspectrogram


using ControlSystems, DSP, Plots, Juno#, Convex, FirstOrderSolvers
using Test

using DiffResults
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
    ll::Float64
end
function KalmanModel(At::Array{T,3},Bt::Array{T,3},Pt::Array{T,3},extend::Bool) where T
    if extend
        At = cat(At,At[:,:,end], dims=3)
        Bt = cat(Bt,Bt[:,:,end], dims=3)
        Pt = cat(Pt,Pt[:,:,end], dims=3)
    end
    return KalmanModel{T}(At,Bt,Pt,extend,0.0)
end

KalmanModel(At,Bt,Pt,extend::Bool=false) = KalmanModel{eltype(At)}(At,Bt,Pt,extend,0.0)

mutable struct GMMModel <: AbstractModel
    M
    dynamics
    T
end

mutable struct KalmanAR{T} <: LinearTimeVaryingModelsBase.LTVModel
    θ::Array{T,2}
    Pt::Array{T,3}
    extended::Bool
    ll::Float64
end

Base.length(m::KalmanAR) = size(m.θ,2)

include("utilities.jl")
include("peakdetection.jl")
include("statespace_fit.jl")
include("seg_bellman.jl")
include("kalmanmodel.jl")
# include("gmmmodel.jl") # Not yet working with julia v0.6
include("statespace_utils.jl")
include("wrappers.jl")
end # module
