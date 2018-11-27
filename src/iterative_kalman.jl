using LTVModels, LowLevelParticleFilters
using LTVModels, Plots, StatPlots, LinearAlgebra, Distributions
T = 1000
eye(n) = Matrix{Float64}(I,n,n)
σ_state_drift=0.01
σ_param_drift = 0.001
σ_noise = 0.1
A,B,x,u,n,m,N = LTVModels.testdata(T=T, σ_state_drift=σ_state_drift, σ_param_drift=σ_param_drift, seed=1)
xt,ut,y = x[:,1:end-1],u[:,1:end-1],x[:,2:end]
R1          = σ_param_drift^2*eye(n^2+n*m)
R2          = σ_state_drift^2*eye(n)
P0          = 1000000eye(n^2+n*m)

xn = x + σ_noise*randn(size(x))
xn[:,1:2] = x[:,1:2]

vv(x) = [x[:,i] for i = 1:size(x,2)]

function iterative_kalman(xi,u,R1,R2,P0; iters=10, kwargs...)
    x = deepcopy(xi)
    local model
    for i = 1:iters
        model = KalmanModel(x,u,R1,R2,P0; kwargs...)
        kf = KalmanFilter(model.At, model.Bt, I, 0, R2, σ_noise*eye(n), MvNormal(1R2))
        xv,R = smooth(kf, vv(u), vv(x))
        # @show R[1], R[end]
        @show sum(cond(R) for R in R)/T
        x[:,3:end] = hcat(xv[2:end-1]...)
    end
    x,model
end


xh, model = iterative_kalman(xn,u,R1,R2,P0, extend=true, iters=10)

plot(x', c=:blue, layout=3)
plot!(xn', c=:red)
plot!(xh', c=:green)

plot((x-xn)', c=:blue, layout=3, lab="Original noise")
plot!((x-xh)', c=:green, lab="Estimation error")
