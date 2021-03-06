@info "using LTVModels"
using LTVModels
@info "using Test, LinearAlgebra, Statistics, Random"
using Test, LinearAlgebra, Statistics, Random
@info "using Plots, Optim"
using Plots, Optim

eye(n) = Matrix{Float64}(I,n,n)
@testset "LTVModels" begin
    @info "Testing LTVModels"

    @testset "findpeaks" begin
        @info "Testing findpeaks"


        y = randn(1000)
        peaks = LTVModels.findpeaks(y,doplot=isinteractive(), filterlength=1)
        for p in peaks
            @test y[p] >= y[max(1,p-1)]
            @test y[p] >= y[min(length(y),p+1)]
        end


        y = randn(1000)
        peaks = LTVModels.findpeaks(y,doplot=isinteractive(), filterlength=1, minh=2)
        for p in peaks
            @test y[p] >= 2
            @test y[p] >= 2
        end

        y = randn(1000)
        peaks = LTVModels.findpeaks(y,doplot=isinteractive(), filterlength=1, minw=10)
        dpeaks = diff(peaks)
        @test all(>=(10), dpeaks)

    end


    @testset "KalmanModel" begin
        @info "Testing KalmanModel"

        T = 10000
        A,B,d,n,m,N = LTVModels.testdata(T=T, σ_state_drift=0.001, σ_param_drift=0.001)

        R1          = 0.001*eye(n^2+n*m) # Increase for faster adaptation
        R2          = 10*eye(n)
        P0          = 10000R1
        @time model = KalmanModel(d,R1,R2,P0,extend=true)

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
        @time model = KalmanModel(model, prior, d,R1,R2,P0,extend=true)
        errorAp = [norm(A[:,:,t]-model.At[:,:,t]) for t = 1:T]
        errorBp = [norm(B[:,:,t]-model.Bt[:,:,t]) for t = 1:T]
        @static isinteractive() && plot!([errorAp errorBp], lab=["errAp" "errBp"], show=true, subplot=1)

        @test all(errorAp .<= errorA) # We expect this since ground truth was used as prior
        @test all(errorBp .<= errorB)


        @testset "No control signal" begin
            @info "Testing No control signal"
            A,B,d,n,m,N = LTVModels.testdata(T=T, σ_state_drift=0.1, σ_param_drift=0.0003, σ_control=0)
            m = 0
            d = iddata(d.x,zeros(0,length(d)), d.x)
            R1          = 0.0003*eye(n^2) # Increase for faster adaptation
            R2          = 1*eye(n)
            P0          = 10000R1
            D = 1
            for D in [1,2]
                model = KalmanModel(d,R1,R2,P0,extend=true, D=D)

                normA  = [norm(A[:,:,t]) for t                = 1:T]
                errorA = [norm(A[:,:,t]-model.At[:,:,t]) for t = 1:T]

                @test sum(normA) > 4sum(errorA)
                @static isinteractive() && plot([normA errorA], lab=["normA" "errA"], show=false, layout=2, subplot=1, size=(1500,900))#, yscale=:log10)
                @static isinteractive() && plot!(flatten(A), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients", lab="True",subplot=2, c=:red)
                @static isinteractive() && plot!(flatten(model.At), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients", lab="Estimated", subplot=2, c=:blue)
            end

        end

        @testset "LTVAutoRegressive" begin
            @info "Testing LTVAutoRegressive"
            ## Generate chirp signal
            T = 2^14
            e = 0.01randn(T)
            fs = 16_000
            t = range(0,step=1/fs, length=T)
            chirp_f1 = LinRange(2000, 2500, T)
            chirp_f2 = 5000 .+ 200sin.(2pi/T .* (1:T))
            y = sin.(2pi .* chirp_f1 .* t )
            y .+= sin.(2pi .* chirp_f2 .* t )
            yn = y + e
            d = iddata(yn)

            n  = 6
            R1 = eye(n) # Increase for faster adaptation
            R2 = [1e5]
            P0 = 1e4R1
            @time model = LTVAutoRegressive(d,R1,R2,P0,extend=true, D=1)
            RS = LTVModels.rootspectrogram(model,fs)
            @test mean(minimum(abs, RS .-  LinRange(2000, 3000, T), dims=2)) < 20 # for some reason, the increase in frequency is double that of the increase in the chirp. This seems correct as a welch spectrogam shows the same.

            @time model = LTVAutoRegressive(d,R1,R2,P0,extend=true, D=2)
            RS = LTVModels.rootspectrogram(model,fs)
            @test mean(minimum(abs, RS .-  LinRange(2000, 3000, T), dims=2)) < 20


        end

    end


    # Tests ========================================================================

    # Iterative solver =============================================================
    @testset "Statespace fit" begin
        @info "Testing Statespace fit"


        # Generate data

        T_       = 400
        x,xm,u,n,m = LTVModels.testdata(T_)

        # model, cost, steps = fit_statespace_gd(xm,u,20, normType = 1, D = 1, lasso = 1e-8, step=5e-3, momentum=0.99, iters=100, reduction=0.1, extend=true);
        # Profile.clear()

        function callback(k)
            model = LTVModels.statevec2model(SimpleLTVModel,k,n,m,true)
            @static isinteractive() && plot(flatten(model.At), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients", show=true)
        end

        d = iddata(x,u,x)
        dn = iddata(xm,u,xm)
        @time model = LTVModels.fit_admm(SimpleLTVModel(dn,extend=true), dn,17,
        iters    = 20000,
        D        = 1,
        zeroinit = true,
        tol      = 1e-5,
        ridge    = 0,
        cb       = callback);
        # using ProfileView
        # ProfileView.view()
        # model, cost, steps = fit_statespace_gd!(model,xm,u,100, normType = 1, D = 1, lasso = 1e-8, step=5e-3, momentum=0.99, iters=1000, reduction=0.1, extend=true);
        y = predict(model,d);
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
        fit_statespace_constrained(dn,changepoints)


        Random.seed!(0)
        T_       = 200
        x,xm,u,n,m = LTVModels.testdata(T_)
        d = iddata(x,u,x)
        dn = iddata(xm,u,xm)


        # TODO: reenable tests below when figured out error with DiffBase
        model2, res = fit_statespace_gd(dn,5000, normType = 1, D = 2, step=0.01, iters=1000, reduction=0.1, extend=true, opt=LBFGS(m=50));
        y2 = predict(model2,d);
        At2,Bt2 = model2.At,model2.Bt
        e2 = x[:,2:end] - y2[:,1:end-1]
        println("RMS error: ",rms(e2))
        @static isinteractive() && plot(flatten(At2), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients")
        @static isinteractive() && plot!([1,T_÷2-1], [0.95 0.1; 0 0.95][:]'.*ones(2), ylims=(-0.1,1), l=(:dash,:black, 1))
        @static isinteractive() && plot!([T_÷2,T_], [0.5 0.05; 0 0.5][:]'.*ones(2), l=(:dash,:black, 1), grid=false)


        model2, res = fit_statespace_gd(dn,1e5, normType = 2, D = 1, step=0.01, momentum=0.99, iters=10000, reduction=0.01, extend=true, lasso=1e-4);
        y2 = predict(model2,d);
        At2,Bt2 = model2.At,model2.Bt
        e2 = x[:,2:end] - y2[:,1:end-1]
        println("RMS error: ",rms(e2))
        @static isinteractive() && plot(flatten(At2), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients")
        @static isinteractive() && plot!([1,T_÷2-1], [0.95 0.1; 0 0.95][:]'.*ones(2), ylims=(-0.1,1), l=(:dash,:black, 1))
        @static isinteractive() && plot!([T_÷2,T_], [0.5 0.05; 0 0.5][:]'.*ones(2), l=(:dash,:black, 1), grid=false)

        @test norm(rms(e2)) < 2
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


@testset "bellman" begin
    @info "Testing bellman"
    LTVModels.benchmark_const(100, 2, true) # Dynamic Programming Bellman
    LTVModels.benchmark_ss(100, 2, true)    # Dynamic Programming Bellman
    LTVModels.benchmark_lin(100, 2, true)
end

end
