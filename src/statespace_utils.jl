import Base.length
export plot_coeffs,plot_coeffs!,plot_eigvals


length(m::LTVStateSpaceModel) = size(m.At,3)
"""
x' = predict(model, x, u)
x' = predict(model, x, u, t)

Form one-step prediction. If model is an LTVmodel, `x,u` and `model` must have the same length.
If `t` is provided, use model at time `t` to predict a single output only.
"""
function predict(model::LTVStateSpaceModel, x, u)
    n,T  = size(x)
    @assert T<=length(model) "Can not predict further than the number of time steps in the model"
    xnew = Array{eltype(x)}(undef,n,T)
    @views for t = 1:T
        xnew[:,t] = (model.At[:,:,t] * x[:,t] + model.Bt[:,:,t] * u[:,t])'
    end
    xnew
end

function predict(model::LTVStateSpaceModel, x)
    n,T  = size(x)
    @assert T<=length(model) "Can not predict further than the number of time steps in the model"
    xnew = Array{eltype(x)}(undef,n,T)
    @views for t = 1:T
        xnew[:,t] = model.At[:,:,t] * x[:,t]
    end
    xnew
end

function predict(model::LTVStateSpaceModel, x, u, i)
    # @assert i<=length(model) "Can not predict further than the number of time steps in the model"
    xnew = model.At[:,:,i] * x + model.Bt[:,:,i] * u
end

"""
x' = simulate(model, x0, u)

Simulate model forward in time from initial state `x0`. If model is an LTVmodel, `u` and `model` must have the same length.
"""
function simulate(model::LTVStateSpaceModel, x0, u)
    T = size(u,2)
    n = size(model.At,1)
    @assert T > n "The calling convention for u is that time is the second dimention (n,T = size(u))"
    @assert T<=length(model) "Can not simulate further than the number of time steps in the model"
    x = Array{eltype(u)}(undef,n,T)
    x[:,1] = x0
    @views for t = 1:T-1
        x[:,t+1] = model.At[:,:,t] * x[:,t] + model.Bt[:,:,t] * u[:,t]
    end
    x
end

function df(model::LTVStateSpaceModel, x, u)
    fx  = model.At
    fu  = model.Bt
    fxx = []
    fxu = []
    fuu = []
    return fx,fu,fxx,fxu,fuu
end

function plot_eigvals(model::LTVModels.LTVStateSpaceModel, plot_circle=true)
    scatter(hcat([eigvals(model.At[:,:,t]) for t=1:length(model)]...)', title="Eigenvalues of dynamics matrix")
    if plot_circle
        ϕ = range(0, stop=2π, length=100)
        plot!(cos.(ϕ),sin.(ϕ), l=(:black,1))
    end
end

function plot_coeffs!(model::LTVStateSpaceModel; kwargs...)
    n,T = size(model.At,1,3)
    plot!(reshape(model.At,n^2,T)'; title="Coefficients of dynamics matrix", kwargs...)
end

function plot_coeffs(model::LTVStateSpaceModel; kwargs...)
    fig = plot()
    plot_coeffs!(model; kwargs...)
    fig
end
