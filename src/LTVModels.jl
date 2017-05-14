module LTVModels

using DSP, Plots

using ReverseDiff: GradientTape, GradientConfig,GradientResult, gradient!
import DiffBase

abstract type Model end

abstract type LTVModel <: Model end

abstract type LTVStateSpaceModel <: LTVModel end

type SimpleLTVModel{T} <: LTVStateSpaceModel
    At::Array{T,3}
    Bt::Array{T,3}
end
function SimpleLTVModel(At,Bt,extend)
    if extend
        At = cat(3,At,At[:,:,end])
        Bt = cat(3,Bt,Bt[:,:,end])
    end
    SimpleLTVModel(At,Bt)
end

type KalmanModel{T} <: LTVStateSpaceModel
    At::Array{T,3}
    Bt::Array{T,3}
    Pt::Array{T,3}
end

function KalmanModel(At,Bt,Pt,extend)
    if extend
        At = cat(3,At,At[:,:,end])
        Bt = cat(3,Bt,Bt[:,:,end])
        Pt = cat(3,Pt,Pt[:,:,end])
    end
    KalmanModel(At,Bt,Pt)
end

type GMMModel <: Model
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
