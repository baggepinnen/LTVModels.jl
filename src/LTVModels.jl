module LTVModels
using LinearAlgebra, Statistics, Printf, Random, Distributions
using LinearTimeVaryingModelsBase, ControlSystemIdentification
import LinearTimeVaryingModelsBase: AbstractModel, AbstractCost, ModelAndCost, f,
dc,calculate_cost,calculate_final_cost,
predict, simulate, df,costfun, LTVStateSpaceModel,
SimpleLTVModel, rms, sse, nrmse, modelfit, aic

import ControlSystemIdentification: OutputData, InputOutputData, InputOutputStateData, AnyInput, AbstractIdData, oftype, time1, input, output, state
import ControlSystems: ninputs, noutputs, nstates

export predict, simulate, iddata, input, output, state, ninputs, noutputs, nstates

export SimpleLTVModel, KalmanModel, LTVAutoRegressive, rootspectrogram


using ControlSystems, DSP, Plots
using Test

using DiffResults, Optim
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

mutable struct LTVAutoRegressive{T,PT<:Union{Array{T,3},Nothing}} <: LinearTimeVaryingModelsBase.LTVModel
    na::Int
    θ::Array{T,2}
    Pt::PT
    extended::Bool
    ll::Float64
end

Base.length(m::LTVAutoRegressive) = size(m.θ,2)

include("utilities.jl")
include("peakdetection.jl")
include("statespace_fit.jl")
include("seg_bellman.jl")
include("kalmanmodel.jl")
# include("gmmmodel.jl") # Not yet working with julia v0.6
include("statespace_utils.jl")
include("wrappers.jl")
end # module
