import Base.length
export plot_coeffs,plot_coeffs!,plot_eigvals


length(m::LTVStateSpaceModel) = size(m.At,3)

function predict(model::LTVStateSpaceModel, x, u)
    T,n  = size(x)
    @assert T<=length(model) "Can not predict further than the number of time steps in the model"
    xnew = zeros(eltype(x),T,n)
    for t = 1:T
        xnew[t,:] = (model.At[:,:,t] * x[t,:] + model.Bt[:,:,t] * u[t,:])'
    end
    xnew
end

function predict(model::LTVStateSpaceModel, x, u, i)
    # @assert i<=length(model) "Can not predict further than the number of time steps in the model"
    xnew = model.At[:,:,i] * x + model.Bt[:,:,i] * u
end

function simulate(model::LTVStateSpaceModel, x0, u)
    T = size(u,1)
    n = size(model.At,1)
    @assert T<=length(model) "Can not simulate further than the number of time steps in the model"
    x = zeros(eltype(u),T,n)
    x[1,:] = x0
    for t = 1:T-1
        x[t+1,:] = (model.At[:,:,t] * x[t,:] + model.Bt[:,:,t] * u[t,:])'
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
        ϕ = linspace(0,2π,100)
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
