if !isdefined(:AbstractModel)
    include(Pkg.dir("DifferentialDynamicProgramming","src","interfaces.jl"))
end
if !isdefined(:KalmanModel)
    type KalmanModel{T} <: AbstractModel
        A::Array{T,3}
        B::Array{T,3}
        P::Array{T,3}
    end
end

# TODO: introduce directional forgetting!!
function fit_model!(model::KalmanModel, x,u,xnew,R1,R2, extend=false)::KalmanModel
    T,n         = size(x)
    Ta          = extend ? T+1 : T
    m           = size(u,2)
    N           = n^2+n*m
    # R1          = 0.00001*eye(N) # Increase for faster adaptation
    # R2          = 10*eye(n)
    P0          = R1 # Increase for faster initial adaptation
    y           = xnew'
    C           = zeros(n,N,T)
    for t = 1:T
        C[:,:,t] = kron(eye(n),[x[t,:]; u[t,:]]')
    end
    xkn, Pkn    = kalman_smoother(y, C, R1, R2, P0)
    A           = Array{Float64}(n,n,Ta)
    B           = Array{Float64}(n,m,Ta)
    for t = 1:T
        ABt      = reshape(xkn[:,t],n+m,n)'
        A[:,:,t] = ABt[:,1:n]
        B[:,:,t] = ABt[:,n+1:end]
    end
    if extend
        A[:,:,end] = A[:,:,end-1]
        B[:,:,end] = B[:,:,end-1]
        Pkn = cat(3,Pkn, Pkn[:,:,end])
    end
    model.A = A
    model.B = B
    model.P = Pkn
    yhat = predict(model, x,u)
    fit = nrmse(xnew,yhat)
    println("Modelfit: ", round(fit,3))

    return model
end

function fit_model(::Type{KalmanModel}, args...)::KalmanModel
    model = KalmanModel(zeros(1,1,1),zeros(1,1,1),zeros(1,1,1))
    fit_model!(model, args...)
    model
end

function predict(model::KalmanModel, x, u)
    T,n  = size(x)
    xnew = zeros(T,n)
    for t = 1:T
        xnew[t,:] = (model.A[:,:,t] * x[t,:] + model.B[:,:,t] * u[t,:])'
    end
    xnew
end

function predict(model::KalmanModel, x, u, i)
    xnew = model.A[:,:,i] * x + model.B[:,:,i] * u
end

# function predict(model::KalmanModel, batch::Batch)
#
# end


function df(model::KalmanModel, x, u)
    fx  = model.A
    fu  = model.B
    fxx = []
    fxu = []
    fuu = []
    return fx,fu,fxx,fxu,fuu
end

plot_eigvals(model::KalmanModel) = scatter(hcat([eigvals(model.A[:,:,t]) for t=1:T]...)', title="Eigenvalues of dynamics matrix")

function plot_coeffs!(model::KalmanModel; kwargs...)
    n,T = size(model.A,1,3)
    plot!(reshape(model.A,n^2,T)'; title="Coefficients of dynamics matrix", kwargs...)
end

function plot_coeffs(model::KalmanModel; kwargs...)
    fig = plot()
    plot_coeffs!(model; kwargs...)
    fig
end


function forward_kalman(y,C,R1,R2, P0)
    na  = size(C,1)
    n   = size(R1,1)
    ma  = (n-na^2) ÷ na
    sa  = na+ma
    T   = size(y,2)
    xkk = zeros(n,T);
    Pkk = zeros(n,n,T)
    for i = 0:na-1
        ran = (i*sa+1):((i+1)*sa)
        size(C[i+1,ran,:]'), size(y[i+1,:])
        xkk[ran,1]    = C[i+1,ran,:]'\y[i+1,:]  # Initialize to global ls solution
    end
    Pkk[:,:,1]  = P0;
    xk         = xkk
    Pk         = Pkk
    i          = 1
    Ck         = C[:,:,i]
    e          = y[:,i]-Ck*xk[:,i]
    S          = Ck*Pk[:,:,i]*Ck' + R2
    K          = (Pk[:,:,i]*Ck')/S
    xkk[:,i]   = xk[:,i] + K*e
    Pkk[:,:,i] = (I - K*Ck)*Pk[:,:,i]
    for i = 2:T
        Ak         = 1
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
    for i = T-1:-1:1
        Ck          = Pkk[:,:,i]/Pk[:,:,i+1]
        xkn[:,i]    = xkk[:,i] + Ck*(xkn[:,i+1] - xk[:,i+1])
        Pkn[:,:,i]  = Pkk[:,:,i] + Ck*(Pkn[:,:,i+1] - Pk[:,:,i+1])*Ck'
    end
    return xkn, Pkn
end

function test_model_kalman()
    function toOrthoNormal(Ti)
        local T = deepcopy(Ti)
        U_,S_,V_ = svd(T[1:3,1:3])
        local R = U_*diagm([1,1,sign(det(U_*V_'))])*V_'
        T[1:3,1:3] = R
        return T
    end

    n           = 3
    m           = 2
    T           = 10000
    A           = zeros(n,n,T)
    B           = zeros(n,m,T)
    x           = zeros(n,T)
    xnew        = zeros(n,T)
    u           = randn(m,T)
    U,S,V       = toOrthoNormal(randn(n,n)), diagm(0.5rand(n)), toOrthoNormal(randn(n,n))
    A[:,:,1]    = U*S*V'
    B[:,:,1]    = 0.5randn(n,m)
    x[:,1]      = 0.1randn(n)

    for t = 1:T-1
        x[:,t+1]   = A[:,:,t]*x[:,t] + B[:,:,t]*u[:,t] + 0.001randn(n)
        xnew[:,t]  = x[:,t+1]
        A[:,:,t+1] = A[:,:,t] + 0.001randn(n,n)
        B[:,:,t+1] = B[:,:,t] + 0.001randn(n,m)
    end
    R1          = 0.00001*eye(N) # Increase for faster adaptation
    R2          = 10*eye(n)

    model = fit_model(KalmanModel, x',u',xnew',R1,R2,true)

    normA  = [norm(A[:,:,t]) for t                = 1:T]
    normB  = [norm(B[:,:,t]) for t                = 1:T]
    errorA = [norm(A[:,:,t]-model.A[:,:,t]) for t = 1:T]
    errorB = [norm(B[:,:,t]-model.B[:,:,t]) for t = 1:T]
    plot([normA errorA normB errorB], lab=["normA" "errA" "normB" "errB"], show=true)#, yscale=:log10)

end
