using LTVModels, LowLevelParticleFilters
using LTVModels, Plots, StatPlots, LinearAlgebra, Distributions
T = 1000
eye(n) = Matrix{Float64}(I,n,n)
σ_state_drift = 0.0001
σ_param_drift = 0.0001
σ_noise = 1std(x)
A,B,x,u,n,m,N = LTVModels.testdata(T=T, σ_state_drift=σ_state_drift, σ_param_drift=σ_param_drift, seed=1)

R1          = σ_param_drift^2*eye(n^2+n*m)
R2          = σ_state_drift^2*eye(n)
Re          = σ_noise^2*eye(n)

P0          = 1000000000eye(n^2+n*m)

xn = x + σ_noise*randn(size(x))
@show √(mean(abs2, xn-x))
xt,ut,y = xn[:,1:end-1],u[:,1:end-1],x[:,2:end]

vv(x) = [x[:,i] for i = 1:size(x,2)]

function iterated_kalman(xtrue,xi,u,R1,R2,P0; iters=10, kwargs...)
    T = size(xi,2)
    x = deepcopy(xi)
    pe,xe,me,ll,llx = zeros(iters), zeros(iters+1), zeros(iters), zeros(iters), zeros(iters)
    xe[1] = √(mean(abs2, xtrue-x))
    local model
    for i = 1:iters
        model = KalmanModel(x,u,R1,Re,P0; kwargs...)
        kf = KalmanFilter(model.At, model.Bt, I, 0, R2, Re, MvNormal(xi[:,1], Re))
        # xv,R,llxi = smooth(kf, vv(u), vv(x))
        _,xv,R,Rt,llxi = forward_trajectory(kf, vv(u), vv(x))
        # aₘ = 1/(1+iters/10+i)^(0.9)
        aₘ = ((2+iters/10)^(1))/(1+iters/10+i)^(1)
        # aₘ = 1/i
        x .+= aₘ*(hcat(xv...) - x)
        # xv = hcat(xv...)
        pe[i] = √(mean(abs2, y-predict(model,xt,ut)))
        me[i] = √(mean(abs2, A-model.At)) + √(mean(abs2, B-model.Bt))
        xe[i+1] = √(mean(abs2, xtrue-x))
        ll[i] = model.ll
        llx[i] = llxi
    end
    x,model,pe,xe,me,ll,llx
end


xh, model,pe,xe,me,ll,llx = iterated_kalman(x,xn,u,R1,R2,P0, extend=true, iters=3)

# plot(x', c=:blue, layout=3)
# plot!(xn', c=:red)
# plot!(xh', c=:green)

plot((x-xn)', c=:blue, layout=4, lab="Original noise", subplot=1:3)
plot!((x-xh)', c=:green, lab="Estimation error", subplot=1:3)
plot!(pe, subplot=4, lab="Prediction error")
plot!(me, subplot=4, lab="Model error")
# hline!([σ_noise], subplot=4, lab="sigma_e", l=(:dash,:black))
plot!(xe[1:end], subplot=4, lab="State error", yscale=:log10, xscale=:log10, ylims=(NaN,NaN), legend=:topright, legendfontsize=6) |> display

# plot(xe, lab="State error", yscale=:log10, xscale=:log10, xlabel="Iteration")
# plot!(me, lab="Model error")
# # savetikz("/local/home/fredrikb/phdthesis/ltvpaper/figs/iterated_kalman.tex")
#
# plot([ll[2:end] llx[2:end]],xscale=:log10, layout=2)
# # plot!(-llx)
plot(flatten(A), c=:blue, lab="")
plot!(flatten(model.At), c=:red, lab="") |> display

# serialize("/tmp/wtf", (x,xn,xh))



function dlyap(A,B)
    S = randn(size(A))
    for i = 1:100
        S = A*S*A' + B*B'
    end
    S
end


A = A[:,:,1]
B = B[:,:,1]
P = dlyap(A,B)
K = P/(Re+P)
C = K*Re*K'
Ce = cov((x-xh)')

P + A*Re*A'
