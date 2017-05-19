export AbstractModel, AbstractCost, ModelAndCost, @define_modelcost_functions, cost, cost_final, dc,
calculate_cost,calculate_final_cost,
 fit_model, predict, df,cT, LTVStateSpaceModel, SimpleLTVModel, KalmanModel, GMMModel
rms(x)      = sqrt(mean(x.^2))
sse(x)      = x⋅x
nrmse(y,yh) = 100 * (1-rms(y-yh)./rms(y-mean(y)))

# Model interface ====================================
"""
Model interface, implement the following functions\n
see also `AbstractCost`, `ModelAndCost`
```
fit_model(::Type{AbstractModel}, batch::Batch)::AbstractModel

predict(model::AbstractModel, x, u)

function df(model::AbstractModel, x, u, I::UnitRange)
    return fx,fu,fxx,fxu,fuu
end
```
"""
abstract type AbstractModel end

abstract type LTVModel <: AbstractModel end

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

type GMMModel <: AbstractModel
    M
    dynamics
    T
end

"""
    model = fit_model(::Type{AbstractModel}, x,u)::AbstractModel

Fits a model to data
"""
function fit_model(::Type{AbstractModel}, x,u)::AbstractModel
    error("This function is not implemented for your type")
    return model
end

"""
    fit_model!(model::AbstractModel, x,u)::AbstractModel

Refits a model to new data
"""
function fit_model!(model::AbstractModel, x,u)::AbstractModel
    error("This function is not implemented for your type")
    return model
end

"""
    xnew = predict(model::AbstractModel, x, u, i)

Predict the next state given the current state and action
"""
function predict(model::AbstractModel, x, u, i)
    error("This function is not implemented for your type")
    return xnew
end


"""
    fx,fu,fxx,fxu,fuu = df(model::AbstractModel, x, u)

Get the linearized dynamics at `x`,`u`
"""
function df(model::AbstractModel, x, u)
    error("This function is not implemented for your type")
    return fx,fu,fxx,fxu,fuu
end
# Model interface ====================================


# Cost interface ====================================
"""
Cost interface, implement the following functions\n
see also `AbstractModel`, `ModelAndCost`
```
function calculate_cost(::Type{AbstractCost}, x::AbstractVector, u)::Number

function calculate_cost(::Type{AbstractCost}, x::AbstractMatrix, u)::AbstractVector

function calculate_final_cost(::Type{AbstractCost}, x::AbstractVector)::Number

function dc(::Type{AbstractCost}, x, u)
    return cx,cu,cxx,cuu,cxu
end
```
"""
abstract type AbstractCost end

function calculate_cost(c::AbstractCost, x::AbstractVector, u)::Number
    error("This function is not implemented for your type")
    return c
end

function calculate_cost(c::AbstractCost, x::AbstractMatrix, u)::AbstractVector
    error("This function is not implemented for your type")
    return c
end

function calculate_final_cost(c::AbstractCost, x::AbstractVector)::Number
    error("This function is not implemented for your type")
    return c
end

function dc(c::AbstractCost, x, u)
    error("This function is not implemented for your type")
    return cx,cu,cxx,cuu,cxu
end
# Cost interface ====================================


"""
1. Define types that implement the interfaces `AbstractModel` and `AbstractCost`.
2. Create object modelcost = ModelAndCost(model, cost)
3. Run macro @define_modelcost_functions(modelcost). This macro defines the following functions
```
f(x, u, i)  = f(modelcost, x, u, i)
fT(x)       = fT(modelcost, x)
df(x, u, I) = df(modelcost, x, u, I)
```
see also `AbstractModel`, `AbstractCost`
"""
type ModelAndCost
    model::AbstractModel
    cost::AbstractCost
end

function f(modelcost::ModelAndCost, x, u, i)
    predict(modelcost.model, x, u, i)
end

function cT(modelcost::ModelAndCost, x)
    calculate_final_cost(modelcost.cost, x)
end

"""
    fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu = df(modelcost::ModelAndCost, x, u)

Get the linearized dynamics and cost at `x`,`u`
"""
function df(modelcost::ModelAndCost, x, u)
    fx,fu,fxx,fxu,fuu = df(modelcost.model, x, u)
    cx,cu,cxx,cuu,cxu = dc(modelcost.cost, x, u)
    return fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu
end

"""
    define_modelcost_functions(modelcost)
This macro defines the following functions
```
f(x, u, i)  = f(modelcost, x, u, i)
cT(x)       = cT(modelcost, x)
df(x, u)    = df(modelcost, x, u)
```
These functions can only be defined for one type of `ModelAndCost`. If you have several different `ModelAndCost`s, define your functions manually.
see also `ModelAndCost`, `AbstractModel`, `AbstractCost`
"""
macro define_modelcost_functions(modelcost)
    ex = quote
        f(x, u, i)  = LTVModels.f($modelcost, x, u, i)
        cT(x)       = LTVModels.cT($modelcost, x)
        df(x, u)    = LTVModels.df($modelcost, x, u)
    end |> esc
    info("Defined:\nf(x, u, i)  = f($modelcost, x, u, i)\ncT(x) = fT($modelcost, x)\ndf(x, u) = df($modelcost, x, u)")
    return ex
end
