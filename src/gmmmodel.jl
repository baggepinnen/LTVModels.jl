if !isdefined(:AbstractModel)
    include(Pkg.dir("DifferentialDynamicProgramming","src","interfaces.jl"))
end

include("gmmutils.jl")


"""
`M, dynamics, ass, T = fit_model!(model::GMMModel, x, u, y, K [,d1])`

Fits a Gaussian mixture model to a set of data. The objective is to predict `y∈ ℜ N × ny` given `x,u`

`K` is the number of clusters to use

`d1` is the number of dimensions to project the data down to when fitting the model.
"""
function fit_model!(model::GMMModel, x,u, xnew, K;
    d1     = 2size(x,2)+size(u,2),
    doplot = false,
    nTries = 5)::GMMModel

    # Put all data in a large matrix
    Xorig = [x u xnew]
    X, Xmin, Xmax = normalize_data(Xorig, 0.1)
    X     = X'

    D,N   = size(X)
    U,S,V = svd(X)
    X̄     = diagm(S)*V'
    X̄1    = X̄[1:d1,:]
    U1    = U[:,1:d1]
    # X   = [x xnew]
    T     = AffineTransform(U1,vec(Xmin),vec(Xmax))

    # Fit a GMM to determine cluster assignments

    M̄         = fit_good_model(X̄1,K,nTries)
    Q,ass     = get_assignments(M̄, X̄1)

    # Fit locally linear dynamics to data within each cluster
    dynamics = Vector{LinearDynamics}(K)
    for k = 1:K
        dynamics[k] = fit_dynamics(xnew[ass .== k,:], x[ass .== k,:], u[ass .== k,:], λ = 1e-3)
    end

    if doplot
        # Get predictions of the next state using the fitted models
        xp = similar(x)
        # dynamics = Vector{LinearDynamics}(N)
        for n = 1:N
            #     dynamics[n] = conditionalDynamics(M, xp_indices, vec(Xorig[n,1:14]'), dof, T)
            xp[n,:] = predict_dynamics(dynamics[ass[n]], x[n,:], u[n,:])'
        end
        plot(layout=size(x,2))
        plot!(x[2:end,:], c=:blue, lab="State")
        plot!(xp[1:end-1,:], c=:red, lab="Predictions")
        plot!(title="Recorded states and model predictions")

        plot(u,title="Recorded control signals")

        # colors = ["c","b","g","r","m","k","y","w"]
        # matshow(Q);  title("Q - soft assignment matrix")

        scatter3d(X̄1[1,:], X̄1[2,:],X̄1[3,:], zcolor=ass, markersize=10,  title="Points with color coded cluster assignment")

    end
    model.M        = M̄
    model.dynamics = dynamics
    model.T        = T
    yhat = predict(model, x,u)
    fit = nrmse(xnew,yhat)
    println("Modelfit: ", round(fit,3))
    return model
end

function fit_model(::Type{GMMModel}, x, u, y, args...; kwargs...)::GMMModel
    model = GMMModel(0,0,0)
    fit_model!(model, x, u, y, args...; kwargs...)
    model
end


function predict(model::GMMModel,x,u)
    u[isnan(u)] = 0
    X    = [x u]'
    Tr   = strip(model.T,1:size(X,1))
    ass  = get_assignments_robust(model.M, X, Tr)
    T,n  = size(x)
    xnew = zeros(T,n)
    for t = 1:T
        dyn       = model.dynamics[ass[t]]
        xnew[t,:] = dyn.fx*x[t,:] + dyn.fu*u[t,:]
    end
    return xnew
end

function predict(model::GMMModel,x::Vector,u,i)
    n    = length(x)
    X    = [x; u]
    Tr   = strip(model.T,1:size(X,1))
    ass  = get_assignments_robust(model.M, X, Tr)
    xnew = zeros(T,n)
    dyn  = model.dynamics[ass[1]]
    xnew = dyn.fx*x + dyn.fu*u
end

"""
    df(model::GMMModel,x,u)
x ∈ ℜ n × N
u ∈ ℜ m × N
"""
function df(model::GMMModel,x,u)
    u[isnan(u)] = 0
    n,N = size(x)
    m   = size(u,1)
    fx  = Array{Float64}(n,n,N)
    fu  = Array{Float64}(n,m,N)
    X   = [x; u]
    Tr  = strip(model.T,1:size(X,1))
    ass = get_assignments_robust(model.M, X, Tr)
    for i = 1:N
        dyn = model.dynamics[ass[i]]
        fx[:,:,i] = dyn.fx
        fu[:,:,i] = dyn.fu
    end
    fxx=fxu=fuu = []
    return fx,fu,fxx,fxu,fuu
end


function test_gmmmodel()
    function toOrthoNormal(Ti)
        local T = deepcopy(Ti)
        U_,S_,V_ = svd(T[1:3,1:3])
        local R = U_*diagm([1,1,sign(det(U_*V_'))])*V_'
        T[1:3,1:3] = R
        return T
    end

    A,B,x,xnew,u,n,m,N = testdata(T=10000, σ_state_drift=0.001, σ_param_drift=0.001)
    
    model = fit_model(GMMModel, x,u,xnew, 5,doplot = false, nTries = 5, d1 = 6)

    xnewhat = predict(model,x,u)

    plot(xnew, lab="x", show=true, layout=n)#, yscale=:log10)
    plot!(xnewhat, lab="xhat")

    fx,fu = df(model,x',u')[1:2]
    normA  = [norm(A[:,:,t]) for t = 1:T]
    normB  = [norm(B[:,:,t]) for t = 1:T]
    errorA = [norm(A[:,:,t]-fx[:,:,t]) for t = 1:T]
    errorB = [norm(B[:,:,t]-fu[:,:,t]) for t = 1:T]
    plot([normA errorA normB errorB], lab=["normA" "errA" "normB" "errB"], show=true)#, yscale=:log10)

end
