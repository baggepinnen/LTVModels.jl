if !@isdefined(AbstractModel)
    include(Pkg.dir("DifferentialDynamicProgramming","src","interfaces.jl"))
end


# TODO: introduce directional forgetting!!
# TODO: predict with covariacne (filter). This should include both model covariacne and state covariance for kalman models
# TODO: Enable imposing of known structure with e.g. a boolean matrix and a coefficient matrix to tell the algorithm which entries are known to be ==1, ==0, ==h etc.
# Use this matrix to either set some values in C, xkn, Pkn, At,Bt to zero

function KalmanModel(model::KalmanModel, xi,u,R1,R2, P0=100R1; extend=false, printfit=true)::KalmanModel
    x,u,xnew = xi[:,1:end-1],u[:,1:end-1],xi[:,2:end]
    n,T = size(x)
    @assert T > n "The calling convention for x and u is that time is the second dimention"
    Ta  = extend ? T+1 : T
    m   = size(u,1)
    N   = n^2+n*m
    y   = copy(xnew)
    C   = zeros(n,N,T)
    @views for t = 1:T
        C[:,:,t] = kron(eye(n),[x[:,t]; u[:,t]]')
    end
    xkn, Pkn    = kalman_smoother(y, C, R1, R2, P0)
    @views for t = 1:T
        ABt      = reshape(xkn[:,t],n+m,n)'
        model.At[:,:,t] .= ABt[:,1:n]
        model.Bt[:,:,t] .= ABt[:,n+1:end]
    end
    @views if extend # Extend model one extra time step (primitive way)
        model.At[:,:,end] .= model.At[:,:,end-1]
        model.Bt[:,:,end] .= model.Bt[:,:,end-1]
        Pkn = cat(3,Pkn, Pkn[:,:,end])
    end
    model.extended = extend
    model.Pt = Pkn
    if printfit
        yhat = predict(model, x,u)
        fit = nrmse(xnew,yhat)
        println("Modelfit: ", round.(fit,3))
    end

    return model
end

# function KalmanModel(model::KalmanModel, x,u,xnew,R1,R2, P0=100R1; extend=false, printfit=true)::KalmanModel
#     x,u,xnew = xi[:,1:end-1],u[:,1:end-1],xi[:,2:end]
#     n,T = size(x)
#     @assert T > n "The calling convention for x and u is that time is the second dimention"
#     Ta  = extend ? T+1 : T
#     m   = size(u,1)
#     N   = n^2+n*m
#     y   = copy(xnew)
#     C   = zeros(n,N,T)
#     @views for t = 1:T
#         C[:,:,t] = kron(eye(n),[x[:,t]; u[:,t]]')
#     end
#     xkn, Pkn    = kalman_smoother(y, C, R1, R2, P0)
#     @views for t = 1:T
#         ABt      = reshape(xkn[:,t],n+m,n)'
#         model.At[:,:,t] .= ABt[:,1:n]
#         model.Bt[:,:,t] .= ABt[:,n+1:end]
#     end
#     @views if extend # Extend model one extra time step (primitive way)
#         model.At[:,:,end] .= model.At[:,:,end-1]
#         model.Bt[:,:,end] .= model.Bt[:,:,end-1]
#         Pkn = cat(3,Pkn, Pkn[:,:,end])
#     end
#     model.extended = extend
#     model.Pt = Pkn
#     if printfit
#         yhat = predict(model, x,u)
#         fit = nrmse(xnew,yhat)
#         println("Modelfit: ", round.(fit,3))
#     end
#
#     return model
# end


function KalmanModel(model::KalmanModel, prior::KalmanModel, args...; printfit = true, kwargs...)::KalmanModel
    model = KalmanModel(model, args...; printfit = false, kwargs...) # Fit model in the standard way without prior
    n,m,T = size(model.Bt)
    # @views model2statevec(model,t) = [vec(model.At[:,:,t]);  vec(model.Bt[:,:,t])][:]
    # @views model2statevec(model,t) = [model.At[:,:,t]  model.Bt[:,:,t]] |> vec
    @views for t = 1:T     # Incorporate prior
        Pkk               = model.Pt[:,:,t]
        K̄                 = Pkk/(Pkk + prior.Pt[:,:,t])
        model.Pt[:,:,t] .-= K̄*Pkk
        x                 = model2statevec(model,t)
        xp                = model2statevec(prior,t)
        x               .+= K̄*(xp-x)
        ABt               = reshape(x,n+m,n)'
        model.At[:,:,t]  .= ABt[:,1:n]
        model.Bt[:,:,t]  .= ABt[:,n+1:end]

    end
    if printfit
        # yhat = predict(model, x,u)
        # fit = nrmse(xnew,yhat)
        # println("Modelfit: ", round(fit,3))
    end
    model
end



function forward_kalman(y,C,R1,R2, P0)
    na,n  = size(C)
    ma  = (n-na^2) ÷ na
    sa  = na+ma
    T   = size(y,2)
    xkk = zeros(n,T);
    Pkk = zeros(n,n,T)
    for i = 0:na-1 # TODO: This should be an input, maybe only initialize with first n_param datapoints?
        ran = (i*sa+1):((i+1)*sa)
        data_to_use = 1:min(2n, size(y,2))
        xkk[ran,1]    = C[i+1,ran,data_to_use]'\y[i+1,data_to_use]  # Initialize to semi-global ls solution
    end
    copy!(@view(Pkk[:,:,1]), P0)
    xk         = copy(xkk)
    Pk         = copy(Pkk)
    i          = 1
    Ck         = @view C[:,:,i]
    e          = y[:,i]-Ck*xk[:,i]
    S          = Ck*Pk[:,:,i]*Ck' + R2
    K          = (Pk[:,:,i]*Ck')/S
    xkk[:,i]   = xk[:,i] + K*e
    Pkk[:,:,i] = (I - K*Ck)*Pk[:,:,i]
    @views for i = 2:T
        Ak         = 1 # This just assumes a random walk, no additional dynamics
        Ck         = C[:,:,i]
        xk[:,i]    = Ak*xkk[:,i-1]
        Pk[:,:,i]  = Ak*Pkk[:,:,i-1]*Ak' + R1
        e          = y[:,i]-Ck*xk[:,i]
        S          = Ck*Pk[:,:,i]*Ck' + R2
        K          = (Pk[:,:,i]*Ck')/S
        xkk[:,i]   = xk[:,i] + K*e
        Pkk[:,:,i] = (I - K*Ck)*Pk[:,:,i]
    end

    return xkk,xk,Pkk,Pk
end

function kalman_smoother(y, C, R1, R2, P0)
    T             = size(y,2)
    xkk,xk,Pkk,Pk = forward_kalman(y,C,R1,R2, P0)
    xkn           = similar(xkk)
    Pkn           = similar(Pkk)
    xkn[:,end]    = xkk[:,end]
    Pkn[:,:,end]  = Pkk[:,:,end]
    @views for i = T-1:-1:1
        Ck          = Pkk[:,:,i]/Pk[:,:,i+1]
        xkn[:,i]    = xkk[:,i] + Ck*(xkn[:,i+1] - xk[:,i+1])
        Pkn[:,:,i]  = Pkk[:,:,i] + Ck*(Pkn[:,:,i+1] - Pk[:,:,i+1])*Ck'
    end
    return xkn, Pkn
end
