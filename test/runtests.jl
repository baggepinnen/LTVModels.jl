using LTVModels
using Base.Test
using Plots

function test_kalmanmodel(T = 10000)

    A,B,x,u,n,m,N = LTVModels.testdata(T=T, σ_state_drift=0.001, σ_param_drift=0.001)

    R1          = 0.001*eye(n^2+n*m) # Increase for faster adaptation
    R2          = 10*eye(n)
    P0          = 10000R1
    @time model = KalmanModel(copy(x),copy(u),R1,R2,P0,extend=true)

    normA  = [norm(A[:,:,t]) for t                = 1:T]
    normB  = [norm(B[:,:,t]) for t                = 1:T]
    errorA = [norm(A[:,:,t]-model.At[:,:,t]) for t = 1:T]
    errorB = [norm(B[:,:,t]-model.Bt[:,:,t]) for t = 1:T]

    @test sum(normA) > 1.8sum(errorA)
    @test sum(normB) > 10sum(errorB)
    @static isinteractive() && plot([normA errorA normB errorB], lab=["normA" "errA" "normB" "errB"], show=false, layout=2, subplot=1, size=(1500,900))#, yscale=:log10)
    @static isinteractive() && plot!(flatten(A), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients", lab="True",subplot=2, c=:red)
    @static isinteractive() && plot!(flatten(model.At), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients", lab="Estimated", subplot=2, c=:blue)
    P = zeros(N,N,T)
    for i=1:T
        P[:,:,i] .= 0.01eye(N)
    end
    prior = KalmanModel(A,B,P) # Use ground truth as prior
    @time model = KalmanModel(model, prior, copy(x),copy(u),R1,R2,P0,extend=true)
    errorAp = [norm(A[:,:,t]-model.At[:,:,t]) for t = 1:T]
    errorBp = [norm(B[:,:,t]-model.Bt[:,:,t]) for t = 1:T]
    @static isinteractive() && plot!([errorAp errorBp], lab=["errAp" "errBp"], show=true, subplot=1)

    @test all(errorAp .<= errorA) # We expect this since ground truth was used as prior
    @test all(errorBp .<= errorB)
end



# Tests ========================================================================

# Iterative solver =============================================================
function test_fit_statespace()
    # Generate data
# using Revise
# using LTVModels, ProximalOperators
    T_       = 400
    x,xm,u,n,m = LTVModels.testdata(T_)

    # model, cost, steps = fit_statespace_gd(xm,u,20, normType = 1, D = 1, lasso = 1e-8, step=5e-3, momentum=0.99, iters=100, reduction=0.1, extend=true);
    # Profile.clear()

    function callback(m)
        @static isinteractive() && plot(flatten(m.At), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients", show=true)
    end

    @time model = LTVModels.fit_statespace_admm(xm,u,17, extend=true,
        iters    = 20000,
        D        = 1,
        zeroinit = true,
        tol      = 1e-5,
        ridge    = 0,
        cb       = callback);
    # using ProfileView
    # ProfileView.view()
    # model, cost, steps = fit_statespace_gd!(model,xm,u,100, normType = 1, D = 1, lasso = 1e-8, step=5e-3, momentum=0.99, iters=1000, reduction=0.1, extend=true);
    y = predict(model,x,u);
    e = x[:,2:end] - y[:,1:end-1]
    println("RMS error: ",rms(e))

    At,Bt = model.At,model.Bt
    @static isinteractive() && begin
        plot(flatten(At), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients")
        plot!([1,T_÷2-1], [0.95 0.1; 0 0.95][:]'.*ones(2), l=(:dash,:black, 1))
        plot!([T_÷2,T_], [0.5 0.05; 0 0.5][:]'.*ones(2), l=(:dash,:black, 1), grid=false)
        gui()
    end

    # R1          = 0.1*eye(n^2+n*m) # Increase for faster adaptation
    # R2          = 10*eye(n)
    # P0          = 10000R1
    # modelk = KalmanModel(x,u,R1,R2,P0,extend=true)
    # @static isinteractive() && plot!(flatten(modelk.At), l=(2,:auto), lab="Kalman", c=:red)

    # # savetikz("figs/ss.tex", PyPlot.gcf())#, [" axis lines = middle,enlargelimits = true,"])
    #
    # @static isinteractive() && plot(y, lab="Estimated state values", l=(:solid,), xlabel="Time index", ylabel="State value", grid=false, layout=2)
    # @static isinteractive() && plot!(x[2:end,:], lab="True state values", l=(:dash,))
    # @static isinteractive() && plot!(xm[2:end,:], lab="Measured state values", l=(:dot,))
    # # savetikz("figs/states.tex", PyPlot.gcf())#, [" axis lines = middle,enlargelimits = true,"])

    # @static isinteractive() && plot([cost[1:end-1] steps], lab=["Cost" "Stepsize"],  xscale=:log10, yscale=:log10)


    act = activation(model)
    changepoints = [findmax(act)[2]]
    fit_statespace_constrained(xm,u,changepoints)


    # TODO: reenable tests below when figured out error with DiffBase
    # model2, cost2, steps2 = fit_statespace_gd(xm,u,5000, normType = 1, D = 2, step=0.01, iters=10000, reduction=0.1, extend=true);
    # y2 = predict(model2,x,u);
    # At2,Bt2 = model2.At,model2.Bt
    # e2 = x[:,2:end] - y2[:,1:end-1]
    # println("RMS error: ",rms(e2))
    # @static isinteractive() && plot(flatten(At2), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients")
    # @static isinteractive() && plot!([1,T_÷2-1], [0.95 0.1; 0 0.95][:]'.*ones(2), ylims=(-0.1,1), l=(:dash,:black, 1))
    # @static isinteractive() && plot!([T_÷2,T_], [0.5 0.05; 0 0.5][:]'.*ones(2), l=(:dash,:black, 1), grid=false)
    #
    #
    # model2, cost2, steps2 = fit_statespace_gd(xm,u,1e10, normType = 2, D = 2, step=0.01, momentum=0.99, iters=10000, reduction=0.01, extend=true, lasso=1e-4);
    # y2 = predict(model2,x,u);
    # At2,Bt2 = model2.At,model2.Bt
    # e2 = x[:,2:end] - y2[:,1:end-1]
    # println("RMS error: ",rms(e2))
    # @static isinteractive() && plot(flatten(At2), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients")
    # @static isinteractive() && plot!([1,T_÷2-1], [0.95 0.1; 0 0.95][:]'.*ones(2), ylims=(-0.1,1), l=(:dash,:black, 1))
    # @static isinteractive() && plot!([T_÷2,T_], [0.5 0.05; 0 0.5][:]'.*ones(2), l=(:dash,:black, 1), grid=false)

    rms(e)
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

    model = GMMModel(x,u,xnew, 5,doplot = false, nTries = 5, d1 = 6)

    xnewhat = predict(model,x,u)

    @static isinteractive() && plot(xnew, lab="x", show=true, layout=n)#, yscale=:log10)
    @static isinteractive() && plot!(xnewhat, lab="xhat")

    fx,fu = df(model,x',u')[1:2]
    normA  = [norm(A[:,:,t]) for t = 1:T]
    normB  = [norm(B[:,:,t]) for t = 1:T]
    errorA = [norm(A[:,:,t]-fx[:,:,t]) for t = 1:T]
    errorB = [norm(B[:,:,t]-fu[:,:,t]) for t = 1:T]
    @static isinteractive() && plot([normA errorA normB errorB], lab=["normA" "errA" "normB" "errB"], show=true)#, yscale=:log10)

end



# gr()
# LTVModels.test_gmmmodel() # Not working on 0.6 due to GaussianMixtures.jl
@test all(test_fit_statespace() .< 0.3)
LTVModels.benchmark_const(100, 2, true) # Dynamic Programming Bellman
LTVModels.benchmark_ss(100, 2, true)    # Dynamic Programming Bellman
test_kalmanmodel()
