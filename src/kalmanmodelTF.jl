

# TODO: introduce directional forgetting!!
# TODO: predict with covariacne (filter). This should include both model covariacne and state covariance for kalman models


function fit_model!(model::KalmanModelTF, x,u,na,nb,R1,R2, P0=100R1; extend=false, printfit=true)::KalmanModelTF
    if isa(nb, Number)
        nb = nb*ones(size(u,1))
    end
    xnew,regressor = getARXregressor(x,u, na, nb)
    n,T = size(xnew)
    @assert T > n "The calling convention for x and u is that time is the second dimention"
    Ta  = extend ? T+1 : T
    m   = size(u,1)
    num_params   = size(regressor,2) # Number of parameters
    y   = copy(xnew)
    C   = zeros(n,n*num_params,T)
    for t = 1:T
        C[:,:,t] .= kron(eye(n),regressor[t,:]')
    end
    println(size(y), size(C), size(R1), size(R2))
    xkn, Pkn    = kalman_smoother(y, C, R1, R2, P0)
    model.params = xkn
    @views if extend # Extend model one extra time step (primitive way)
        Pkn = cat(3,Pkn, Pkn[:,:,end])
        model.params[:,end] .= model.At[:,end-1]
    end
    model.Pt = Pkn
    model.extended = extend
    if printfit
        yhat = predict(model, x,u)
        fit = nrmse(xnew,yhat)
        println("Modelfit: ", round.(fit,3))
    end

    return model
end


function fit_model!(model::KalmanModelTF, prior::KalmanModelTF, args...; printfit = true, kwargs...)::KalmanModelTF
    model = fit_model!(model, args...; printfit = false, kwargs...) # Fit model in the standard way without prior
    num_params,T = size(model.params)
    @views for t = 1:T     # Incorporate prior
        Pkk                 = model.Pt[:,:,t]
        K̄                   = Pkk/(Pkk + prior.Pt[:,:,t])
        model.Pt[:,:,t]   .-= K̄*Pkk
        x                   = model.params[:,t]
        xp                  = prior.params[:,t]
        x                 .+= K̄*(xp-x)
        model.params[:,t]  .= x
    end
    if printfit
        yhat = predict(model, x,u)
        fit = nrmse(xnew,yhat)
        println("Modelfit: ", round(fit,3))
    end
    model
end




function test_kalmanmodelTF(T = 10000)

    # using LTVModels
    T = 100
    A,B,x,u,n,m,N,tfs = LTVModels.testdataTF(;T=T, σ_state_drift=0.001, σ_param_drift=0.001)

    na = 2
    nb = [2,2]
    N = n*(sum(nb)+n*na)
    R1          = 0.001*eye(N) # Increase for faster adaptation
    R2          = 10*eye(n)
    P0          = 10000R1
    @time model = fit_model(KalmanModelTF, copy(x),copy(u),na,nb,R1,R2,P0;extend=true)


end

# @time model = fit_model(KalmanModelTF, copy(x),copy(u),na,nb,R1,R2,P0;extend=true)
