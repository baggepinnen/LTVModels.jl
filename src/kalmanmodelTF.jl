

# TODO: introduce directional forgetting!!
# TODO: predict with covariacne (filter). This should include both model covariacne and state covariance for kalman models

struct KalmanModelTF{T1,T2}
    params::T1
    Pt::T2
    extended::Bool
end

function fit_model!(model::KalmanModelTF, x,u,na,nb,R1,R2, P0=100R1; extend=false, printfit=true)::KalmanModelTF
    if isa(Number, na)
        na = na*ones(size(u,1)
    end
    xnew,regressor = getARXregressor(x,u, na, nb)
    n,T = size(xnew)
    @assert T > n "The calling convention for x and u is that time is the second dimention"
    Ta  = extend ? T+1 : T
    m   = size(u,1)
    num_params   = na+sum(nb) # Number of parameters
    y   = copy(xnew)
    C   = zeros(n,n*num_params,T)
    for t = 1:T
        C[:,:,t] .= kron(eye(n),regressor[t,:]')
    end
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
    # @views model2statevec(model,t) = [vec(model.At[:,:,t]);  vec(model.Bt[:,:,t])][:]
    # @views model2statevec(model,t) = [model.At[:,:,t]  model.Bt[:,:,t]] |> vec
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

    A,B,x,u,n,m,N = LTVModels.testdata(T=T, σ_state_drift=0.001, σ_param_drift=0.001)

    R1          = 0.001*eye(n^2+n*m) # Increase for faster adaptation
    R2          = 10*eye(n)
    P0          = 10000R1
    @time model = fit_model(KalmanModelTF, copy(x),copy(u),R1,R2,P0,extend=true)

    normA  = [norm(A[:,:,t]) for t                = 1:T]
    normB  = [norm(B[:,:,t]) for t                = 1:T]
    errorA = [norm(A[:,:,t]-model.At[:,:,t]) for t = 1:T]
    errorB = [norm(B[:,:,t]-model.Bt[:,:,t]) for t = 1:T]

    @test sum(normA) > 1.8sum(errorA)
    @test sum(normB) > 10sum(errorB)
    plot([normA errorA normB errorB], lab=["normA" "errA" "normB" "errB"], show=false, layout=2, subplot=1, size=(1500,900))#, yscale=:log10)
    plot!(flatten(A), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients", lab="True",subplot=2, c=:red)
    plot!(flatten(model.At), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients", lab="Estimated", subplot=2, c=:blue)
    P = zeros(N,N,T)
    for i=1:T
        P[:,:,i] .= 0.01eye(N)
    end
    prior = KalmanModelTF(A,B,P) # Use ground truth as prior
    @time model = fit_model!(model, prior, copy(x),copy(u),R1,R2,P0,extend=true)
    errorAp = [norm(A[:,:,t]-model.At[:,:,t]) for t = 1:T]
    errorBp = [norm(B[:,:,t]-model.Bt[:,:,t]) for t = 1:T]
    plot!([errorAp errorBp], lab=["errAp" "errBp"], show=true, subplot=1)

    @test all(errorAp .<= errorA) # We expect this since ground truth was used as prior
    @test all(errorBp .<= errorB)
end
