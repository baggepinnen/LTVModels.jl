using Revise
using LTVModels
T_       = 1000
x,xm,u,n,m = LTVModels.testdata(T_)

model, cost, steps = fit_statespace_gd(xm,u,20, normType = 1, D = 1, lasso = 1e-8, step=5e-3, momentum=0.99, iters=100, reduction=0.1, extend=true);

model = fit_statespace_jump!(model, xm,u,20, normType = 1, D = 1, lasso = 1e-8, extend=true);

function callback(x)
    k = reshape(x,n*(n+m),T_-1)'
    model = LTVModels.statevec2model(k,n,m,true)
    At,Bt = model.At,model.Bt

    plot(flatten(At), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients", show=false)
    plot!([1,T_÷2-1], [0.95 0.1; 0 0.95][:]'.*ones(2), ylims=(-0.1,1), l=(:dash,:black, 1), show=false)
    plot!([T_÷2,T_], [0.5 0.05; 0 0.5][:]'.*ones(2), l=(:dash,:black, 1), grid=false, yaxis=(-0.1,1.1));gui()
end

model = LTVModels.fit_statespace_admm(model, xm,u,20,
                normType   = 1,
                D          = 1,
                lasso      = 1e-8,
                tol        = 4e-4,
                iters      = 20_000,
                cb         = callback,
                printerval = 20,
                extend     = true);

# strace = @stacktrace (JuMP,ProximalOperators,LTVModels) fit_statespace_jump!(model, xm,u,20, normType = 1, D = 1, lasso = 1e-8, extend=true);



y = predict(model,x,u);
e = x[:,2:end] - y[:,1:end-1]
At,Bt = model.At,model.Bt
println("RMS error: ",rms(e))

plot(flatten(At), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients")
plot!([1,T_÷2-1], [0.95 0.1; 0 0.95][:]'.*ones(2), ylims=(-0.1,1), l=(:dash,:black, 1))
plot!([T_÷2,T_], [0.5 0.05; 0 0.5][:]'.*ones(2), l=(:dash,:black, 1), grid=false);gui()
