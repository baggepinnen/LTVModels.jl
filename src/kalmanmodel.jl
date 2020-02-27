if !@isdefined(AbstractModel)
    include(Pkg.dir("DifferentialDynamicProgramming","src","interfaces.jl"))
end


"""
    forward_kalman(y, C, R1, R2, P0)

#Arguments:
- `y`: DESCRIPTION
- `C`: DESCRIPTION
- `R1`: State noise
- `R2`: Meas noise
- `P0`: Initial state cov
"""
function forward_kalman(y,C,R1,R2, P0)
    na,n  = size(C)
    ma  = (n-na^2) ÷ na
    sa  = na+ma
    T   = size(y,2)
    xkk = zeros(n,T);
    Pkk = zeros(n,n,T)
    for i = 0:na-1 # TODO: This should be an input, maybe only initialize with first n_param datapoints?
        ran = (i*sa+1):((i+1)*sa)
        data_to_use = 1:min(10n, size(y,2))
        xkk[ran,1]    = C[i+1,ran,data_to_use]'\y[i+1,data_to_use]  # Initialize to semi-global ls solution
    end
    R2d = MvNormal(R2)
    Pkk[:,:,1] .= P0
    xk         = copy(xkk)
    Pk         = copy(Pkk)
    i          = 1
    Ck         = C[:,:,i]
    e          = y[:,i]-Ck*xk[:,i]
    ll         = logpdf(R2d,e)
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
        ll        += logpdf(R2d,e)
        S          = Ck*Pk[:,:,i]*Ck' + R2
        K          = (Pk[:,:,i]*Ck')/S
        xkk[:,i]   = xk[:,i] + K*e
        Pkk[:,:,i] = (I - K*Ck)*Pk[:,:,i]
    end

    return xkk,xk,Pkk,Pk,ll
end

function kalman_smoother(y, C, R1, R2, P0)
    T                = size(y,2)
    xkk,xk,Pkk,Pk,ll = forward_kalman(y,C,R1,R2, P0)
    xkn              = similar(xkk)
    Pkn              = similar(Pkk)
    xkn[:,end]       = xkk[:,end]
    Pkn[:,:,end]     = Pkk[:,:,end]
    @views for i = T-1:-1:1
        Ck          = Pkk[:,:,i]/Pk[:,:,i+1]
        xkn[:,i]    = xkk[:,i] + Ck*(xkn[:,i+1] - xk[:,i+1])
        Pkn[:,:,i]  = Pkk[:,:,i] + Ck*(Pkn[:,:,i+1] - Pk[:,:,i+1])*Ck'
    end
    return xkn, Pkn,ll
end


# TODO: Enable imposing of known structure with e.g. a boolean matrix and a coefficient matrix to tell the algorithm which entries are known to be ==1, ==0, ==h etc.
# Use this matrix to either set some values in C, xkn, Pkn, At,Bt to zero
eye(n) = Matrix{Float64}(I,n,n)


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
    xkn, Pkn,ll    = kalman_smoother(y, C, R1, R2, P0)
    model.ll = ll
    @views for t = 1:T
        ABt      = reshape(xkn[:,t],n+m,n)'
        model.At[:,:,t] .= ABt[:,1:n]
        model.Bt[:,:,t] .= ABt[:,n+1:end]
    end
    @views if extend # Extend model one extra time step (primitive way)
        model.At[:,:,end] .= model.At[:,:,end-1]
        model.Bt[:,:,end] .= model.Bt[:,:,end-1]
        Pkn = cat(Pkn, Pkn[:,:,end], dims=3)
    end
    model.extended = extend
    model.Pt = Pkn
    if printfit
        yhat = predict(model, x,u)
        fit = nrmse(xnew,yhat)
        println("Modelfit: ", round.(fit,digits=3))
    end

    return model
end

function KalmanModel(model::KalmanModel, prior::KalmanModel, X,U,args...; printfit = true, kwargs...)::KalmanModel
    model = KalmanModel(model, X,U,args...; printfit = false, kwargs...) # Fit model in the standard way without prior
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
        yhat = predict(model, X[:,1:end-1],U[:,1:end-1])
        fit = nrmse(X[:,2:end],yhat)
        println("Modelfit: ", round.(fit,sigdigits=3))
    end
    model
end



"""
    KalmanModel(xi, R1, R2, P0=100R1; extend=false, D=1, printfit=true)::KalmanModel

Estimate a Kalman model without control input

#Arguments:
- `xi`: states
- `R1`: Parameter transition noise
- `R2`: State transition noise
- `P0`: Initial state cov
- `extend`: add one data point at the end to have length of parameter vector match length of input.
- `D`: Order of integration of the noise.
"""
function KalmanModel(model::KalmanModel, xi,R1,R2, P0=100R1; extend=false, printfit=true, D=1)::KalmanModel
    x,xnew = xi[:,1:end-1],xi[:,2:end]
    n,T = size(x)
    @assert T > n "The calling convention for x and u is that time is the second dimention"
    Ta  = extend ? T+1 : T
    N   = n^2
    y   = copy(xnew)
    C   = zeros(n,D*N,T)
    if D == 2
        R1 = kron([1/4 1/2; 1/2 1]+0.001I, R1)
        P0 = kron([1/4 1/2; 1/2 1]+0.001I, P0)
    end
    @views for t = 1:T
        C[:,1:N,t] = kron(eye(n),x[:,t]')
    end
    xkn, Pkn,ll    = kalman_smoother(y, C, R1, R2, P0)
    model.ll = ll
    @views for t = 1:T
        model.At[:,:,t] .= reshape(xkn[1:N,t],n,n)'
    end
    @views if extend # Extend model one extra time step (primitive way)
        model.At[:,:,end] .= model.At[:,:,end-1]
        Pkn = cat(Pkn, Pkn[:,:,end], dims=3)
    end
    model.extended = extend
    model.Pt = Pkn
    if printfit
        yhat = predict(model, x)
        fit = nrmse(xnew,yhat)
        println("Modelfit: ", round.(fit,digits=3))
    end

    return model
end


function KalmanAR(model::KalmanAR, yi,R1,R2, P0=100R1; extend=false, printfit=true, D=1)
    y,ynew = yi[1:end-1],yi[2:end]
    T = length(y)
    n = size(R1,1)
    Ta  = extend ? T+1 : T
    ϕ   = zeros(1,D*n,T)
    if D == 2
        R1 = kron([1/4 1/2; 1/2 1]+0.001I, R1)
        P0 = kron([1/4 1/2; 1/2 1]+0.001I, P0)
    end
    inds = 1:n
    y0 = [zeros(n-1);y]
    @views for t = 1:T
        ϕ[1,1:n,t] = y0[inds .+ (t-1)]
    end
    xkn, Pkn,ll = kalman_smoother(ynew', ϕ, R1, R2, P0)
    model.ll = ll
    @views for t = 1:T
        model.θ[:,t] .= xkn[1:n,t]
    end
    @views if extend # Extend model one extra time step (primitive way)
        model.θ[:,end] .= model.θ[:,end-1]
        Pkn = cat(Pkn, Pkn[:,:,end], dims=3)
    end
    model.extended = extend
    model.Pt = Pkn
    if printfit
        yhat = predict(model, y)
        fit = nrmse(ynew,yhat)
        println("Modelfit: ", round.(fit,digits=3))
    end

    return model
end



function predict(model::KalmanAR, y)
    T  = length(y)
    n = size(model.θ, 1)
    @assert T<=length(model) "Can not predict further than the number of time steps in the model"
    y0 = [zeros(n-1);y]
    ynew = zeros(eltype(y), T)
    inds = 1:n
    @views for t = 1:T
        ynew[t] = model.θ[:,t]'y0[inds .+ (t-1)]
    end
    ynew
end


function rootspectrogram(model, fs)
    roots = map(1:length(model)) do i
        sys = tf(1,[1;-reverse(model.θ[:,i])],1)
        (sort(pole(sys), by=imag, rev=true)[1:end÷2])
    end
    S = reduce(hcat,roots)
    fs/(2pi) .* angle.(S)'
end
