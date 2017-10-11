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

using ReverseDiff: GradientTape, GradientConfig,GradientResult, gradient!
import DiffBase

mutable struct KalmanModel{T} <: LTVStateSpaceModel
    At::Array{T,3}
    Bt::Array{T,3}
    Pt::Array{T,3}
    extended::Bool
    function KalmanModel{T}(At::Array{T,3},Bt::Array{T,3},Pt::Array{T,3},extend::Bool)
        if extend
            At = cat(3,At,At[:,:,end])
            Bt = cat(3,Bt,Bt[:,:,end])
            Pt = cat(3,Pt,Pt[:,:,end])
        end
        return new(At,Bt,Pt,extend)
    end
end

KalmanModel(At,Bt,Pt,extend::Bool=false) = KalmanModel{eltype(At)}(At,Bt,Pt,extend)

mutable struct GMMModel <: AbstractModel
    M
    dynamics
    T
end

include("utilities.jl")
include("peakdetection.jl")
include("statespace_fit.jl")
include("seg_bellman.jl")
include("kalmanmodel.jl")
# include("gmmmodel.jl") # Not yet working with julia v0.6
include("statespace_utils.jl")
include("wrappers.jl")
end # module
