using LTVModels
using Base.Test


# Iterative solver =============================================================
function test_fit_statespace(lambda)
    # Generate data
    srand(1)
    D        = 1
    normType = 1
    T_       = 400
    n        = 2
    At_      = [0.95 0.1; 0 0.95]
    Bt_      = [0.2; 1]''
    u        = randn(T_)
    x        = zeros(T_,n)
    for t = 1:T_-1
        if t == 200
            At_ = [0.5 0.05; 0 0.5]
        end
        x[t+1,:] = At_*x[t,:] + Bt_*u[t,:] + 0.2randn(n)
    end
    xm = x + 0.2randn(size(x));
    model, cost, steps = fit_statespace_gd(xm,u,lambda, normType = normType, D = D, step=1e-4, momentum=0.992, iters=4000, reduction=0.1, adaptive=true);
    y = predict(x,u,model);
    At,Bt = model.At,model.Bt
    e = x[2:end,:] - y;
    println("RMS error: ",rms(e)^2)

    plot(flatten(At), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients")
    plot!([1,199], [0.95 0.1; 0 0.95][:]'.*ones(2), ylims=(-0.1,1), l=(:dash,:black, 1))
    plot!([200,400], [0.5 0.05; 0 0.5][:]'.*ones(2), l=(:dash,:black, 1), grid=false)
    # # savetikz("figs/ss.tex", PyPlot.gcf())#, [" axis lines = middle,enlargelimits = true,"])
    #
    # plot(y, lab="Estimated state values", l=(:solid,), xlabel="Time index", ylabel="State value", grid=false, layout=2)
    # plot!(x[2:end,:], lab="True state values", l=(:dash,))
    # plot!(xm[2:end,:], lab="Measured state values", l=(:dot,))
    # # savetikz("figs/states.tex", PyPlot.gcf())#, [" axis lines = middle,enlargelimits = true,"])

    # plot([cost[1:end-1] steps], lab=["Cost" "Stepsize"],  xscale=:log10, yscale=:log10)


    activation = segment(At,Bt)
    changepoints = [findmax(activation)[2]]
    At,Bt = fit_statespace_constrained(x,u,changepoints)

    rms(e)
end






# Dynamic programming ==========================================================
function benchmark_const(N, M=1)
    n = 3N
    y = [0.1randn(N); 10+0.1randn(N); 20+0.1randn(N)+linspace(1,10,N)]
    V,t,a = @time seg_bellman(y,M, ones(y))
    tplot = [1;t;n];
    aplot = [a;a[end]];
    plot(y)
    plot!(tplot, aplot, l=:step);gui()
    # yhat, x, a, b
end

function benchmark_ss(T_, M, doplot=false)
    srand(1)
    # M        = 1
    # T_       = 400
    n,m      = 2,1
    At_      = [0.95 0.1; 0 0.95]
    Bt_      = [0.2; 1]''
    u        = randn(T_)
    x        = zeros(T_,n)
    for t = 1:T_-1
        if t == T_÷2
            At_ = [0.5 0.05; 0 0.5]
        end
        x[t+1,:] = At_*x[t,:] + Bt_*u[t,:] + 0.2randn(n)
    end
    xm = x + 0.2randn(size(x));
    input = matrices(xm,u)
    V,t,a = seg_bellman(input,M, ones(T_-1), cost_ss, argmin_ss, doplot=false)
    if doplot
        k = hcat(a...)'
        At = reshape(k[:,1:n^2]',n,n,M+1)
        Bt = reshape(k[:,n^2+1:end]',m,n,M+1)
        At = permutedims(At, [2,1,3])
        Bt = permutedims(Bt, [2,1,3])
        tplot = [1;t;T_];
        plot()
        for i = 1:M+1
            plot!(tplot[i:i+1],flatten(At)[i,:]'.*ones(2), c=[:blue :green :red :magenta], xlabel="Time index", ylabel="Model coefficients")
        end
        plot!([1,199], [0.95 0.1; 0 0.95][:]'.*ones(2), ylims=(-0.1,1), l=(:dash,:black, 1))
        plot!([200,400], [0.5 0.05; 0 0.5][:]'.*ones(2), l=(:dash,:black, 1), grid=false)
    end
    V,t,a
end
# T_vec = round(Int,logspace(2, 4,10))
# times = map(T_vec) do T
#     println(T)
#     benchmark_ss(T, 1)
# end
# plot(T_vec, times)
#
M_vec = [2,4,8,16,32]
times = map(M_vec) do M
    println(M)
    @elapsed benchmark_ss(110, M)
end
plot(M_vec, times)
