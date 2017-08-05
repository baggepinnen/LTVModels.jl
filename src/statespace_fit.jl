export fit_statespace, fit_statespace_gd, fit_statespace_constrained, fit_statespace_gd!, fit_statespace_jump!

# function fit_statespace(x,u,lambda; normType = 2, D = 2, solver=:ECOS, kwargs...)
#     y,A  = matrices(x,u)
#     n,T  = size(x)
#     T   -= 1
#     m    = size(u,1)
#
#     Dx   = getD(D,T)
#     k    = Convex.Variable(T,n^2+n*m)
#
#     loss = 0
#     for i = 1:T
#         ii = (i-1)*n+1
#         ii2 = ii+n-1
#         loss += Convex.sumsquares((y[:,ii:ii2]) - A[ii:ii2,:]*k[i,:]')
#     end
#     loss += lambda*Convex.norm(Dx*k,normType)
#     loss += Convex.sumsquares(k)
#     problem = Convex.minimize(loss)
#     if solver == :ECOS
#         Convex.solve!(problem, ECOS.ECOSSolver(maxit=100, feastol=1e-10, feastol_inacc=1e-5, reltol=1e-09, abstol=1e-09, nitref=50))
#     elseif solver == :Mattias
#         Convex.solve!(problem, FirstOrderSolvers.LongstepWrapper(FirstOrderSolvers.GAPA(),max_iters=5000))
#     else
#         Convex.solve!(problem, SCS.SCSSolver(max_iters=100000, normalize=0, eps=1e-8))
#     end
#
#     At,Bt = ABfromk(k.value,n,m,T)
#     SimpleLTVModel(At,Bt)
# end

function Lcurve(fun, lambdas)
    errors = pmap(fun, lambdas)
    plot(lambdas, errors, xscale=:log10, m=(:cross,))
    errors, lambdas
end

function fit_statespace_gd(x,u,lambda; initializer::Symbol=:kalman, extend=false,kwargs...)
    y,A     = matrices(x,u)
    n,T     = size(x)
    m       = size(u,1)
    T      -= 1
    if initializer != :kalman
        k = A\y
        k = repmat(k',T,1)
        k .+= 0.00001randn(size(k))
        model = statevec2model(k,n,m,false)
    else
        R1 = 0.1*eye(n^2+n*m)
        R2 = 10eye(n)
        P0 = 10000R1
        model = fit_model(KalmanModel, x,u,R1,R2,P0, extend=false)
    end
    fit_statespace_gd!(model, x,u,lambda; extend=extend, kwargs...)
end

function fit_statespace_gd!(model::AbstractModel,x,u,lambda; normType = 1, D = 1, step=0.001, iters=10000, lasso=0, decay_rate=0.999, momentum=0.9, print_period = 100, reduction=0, extend=true, kwargs...)
    if reduction > 0
        decay_rate = decayfun(iters, reduction)
    end
    k = model2statevec(model)
    const y, A     = matrices(x,u)
    nparams = size(A,2)
    n,T     = size(x)
    T      -= 1
    m       = size(u,1)
    diff_fun = D == 2 ? x-> diff(diff(x,1),1) : x-> diff(x,1)
    function lossfun(k2)
        loss    = 0.
        for i = 1:T
            ii = (i-1)*n+1
            ii2 = ii+n-1
            @inbounds @views loss += mean((y[ii:ii2,:] - A[ii:ii2,:]*k2[i,:]).^2)
        end
        NK = length(k2)
        if normType == 1
            loss += lambda^2/NK*sum( sqrt.(sum(diff_fun(k2./std(k2,1)).^2, 2)) )
        else
            loss += lambda/NK*sum(diff_fun(k2).^2)
        end
        if lasso > 0
            loss += lasso^2*sum(abs.(k2))
        end
        loss
    end
    inputs      = (k,)
    loss_tape   = GradientTape(lossfun, inputs)
    results     = similar.(inputs)
    all_results = DiffBase.GradientResult.(results)
    mom         = zeros(k)
    @show bestcost    = lossfun(k)
    costs       = Inf*ones(iters+1); costs[1] = bestcost
    steps       = zeros(iters)
    # opt         = RMSpropOptimizer(k, step, 0.8, momentum)
    opt         = ADAMOptimizer(k; α = step,  β1 = 0.9, β2 = 0.999, ɛ = 1e-8)
    @progress for iter = 1:iters
        steps[iter] = step
        gradient!(all_results, loss_tape, inputs)
        costs[iter] = all_results[1].value
        # mom .= step.*all_results[1].derivs[1] .+ momentum.*mom
        # k .-= mom
        # opt(all_results[1].derivs[1])
        opt(all_results[1].derivs[1], iter)
        if iter % print_period == 0
            println("Iteration: ", iter, " cost: ", round(costs[iter],6), " stepsize: ", step)
        end
        step *=  decay_rate
        opt.α = step
    end

    At,Bt = ABfromk(k,n,m,T)
    SimpleLTVModel{eltype(At)}(At,Bt,extend),costs, steps
end

function fit_statespace_constrained(x,u,changepoints::AbstractVector; extend=true)
    const y,A     = matrices(x,u)
    n,T           = size(x)
    m             = size(u,1)
    nc            = length(changepoints)
    changepointse = [1; changepoints; T-1]
    Ai            = zeros(n,n,nc+1)
    Bi            = zeros(n,m,nc+1)
    k = map(1:nc+1) do i
        ii = (changepointse[i]-1)*n+1
        ii2 = changepointse[i+1]*n-1
        inds = ii:ii2
        k = (A[inds,:]\y[inds])'
    end
    At,Bt = segments2full(k,changepoints,n,m,T)
    return SimpleLTVModel(At,Bt,extend)
end




# Tests ========================================================================

# Iterative solver =============================================================
function test_fit_statespace()
    # Generate data

    T_       = 400
    x,xm,u,n,m = LTVModels.testdata(T_)

    model, cost, steps = fit_statespace_gd(xm,u,20, normType = 1, D = 1, lasso = 1e-8, step=5e-3, momentum=0.99, iters=100, reduction=0.1, extend=true);
    model = fit_statespace_jump!(model, xm,u,20, normType = 1, D = 1, lasso = 1e-8, extend=true);
    # model, cost, steps = fit_statespace_gd!(model,xm,u,100, normType = 1, D = 1, lasso = 1e-8, step=5e-3, momentum=0.99, iters=1000, reduction=0.1, extend=true);
    y = predict(model,x,u);
    e = x[:,2:end] - y[:,1:end-1]
    At,Bt = model.At,model.Bt
    println("RMS error: ",rms(e))

    plot(flatten(At), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients")
    plot!([1,T_÷2-1], [0.95 0.1; 0 0.95][:]'.*ones(2), ylims=(-0.1,1), l=(:dash,:black, 1))
    plot!([T_÷2,T_], [0.5 0.05; 0 0.5][:]'.*ones(2), l=(:dash,:black, 1), grid=false);gui()

    # R1          = 0.1*eye(n^2+n*m) # Increase for faster adaptation
    # R2          = 10*eye(n)
    # P0          = 10000R1
    # modelk = fit_model(KalmanModel, x,u,R1,R2,P0,extend=true)
    # plot!(flatten(modelk.At), l=(2,:auto), lab="Kalman", c=:red)

    # # savetikz("figs/ss.tex", PyPlot.gcf())#, [" axis lines = middle,enlargelimits = true,"])
    #
    # plot(y, lab="Estimated state values", l=(:solid,), xlabel="Time index", ylabel="State value", grid=false, layout=2)
    # plot!(x[2:end,:], lab="True state values", l=(:dash,))
    # plot!(xm[2:end,:], lab="Measured state values", l=(:dot,))
    # # savetikz("figs/states.tex", PyPlot.gcf())#, [" axis lines = middle,enlargelimits = true,"])

    # plot([cost[1:end-1] steps], lab=["Cost" "Stepsize"],  xscale=:log10, yscale=:log10)


    activation = segment(At,Bt)
    changepoints = [findmax(activation)[2]]
    fit_statespace_constrained(x,u,changepoints)


    model2, cost2, steps2 = fit_statespace_gd(xm,u,5000, normType = 1, D = 2, step=0.01, iters=10000, reduction=0.1, extend=true);
    y2 = predict(model2,x,u);
    At2,Bt2 = model2.At,model2.Bt
    e2 = x[:,2:end] - y2[:,1:end-1]
    println("RMS error: ",rms(e2))
    plot(flatten(At2), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients")
    plot!([1,T_÷2-1], [0.95 0.1; 0 0.95][:]'.*ones(2), ylims=(-0.1,1), l=(:dash,:black, 1))
    plot!([T_÷2,T_], [0.5 0.05; 0 0.5][:]'.*ones(2), l=(:dash,:black, 1), grid=false)


    model2, cost2, steps2 = fit_statespace_gd(xm,u,1e10, normType = 2, D = 2, step=0.01, momentum=0.99, iters=10000, reduction=0.01, extend=true, lasso=1e-4);
    y2 = predict(model2,x,u);
    At2,Bt2 = model2.At,model2.Bt
    e2 = x[:,2:end] - y2[:,1:end-1]
    println("RMS error: ",rms(e2))
    plot(flatten(At2), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients")
    plot!([1,T_÷2-1], [0.95 0.1; 0 0.95][:]'.*ones(2), ylims=(-0.1,1), l=(:dash,:black, 1))
    plot!([T_÷2,T_], [0.5 0.05; 0 0.5][:]'.*ones(2), l=(:dash,:black, 1), grid=false)

    rms(e)
end

if false
Profile.clear()
@profile fit_statespace_gd(xm,u,10000, normType = 1, D = D, step=5e-3, momentum=0.99, iters=500, reduction=0.1, extend=true);
ProfileView.view()
end






using JuMP, Ipopt, SCS
function fit_statespace_jump!(model::AbstractModel,x,u,lambda; normType = 1, D = 1,  lasso=0,  extend=true, kwargs...)
    k             = model2statevec(model)
    const y, A    = matrices(x,u)
    nparams       = size(A,2)
    n,T           = size(x)
    T            -= 1
    m             = size(u,1)
    NK            = length(k)
    diff_fun      = D == 2 ? x-> diff(diff(x,1),1) : x-> diff(x,1)
    model         = Model(solver=SCSSolver(
        max_iters = 40000,
        eps       = 1e-5, # convergence tolerance: 1e-3 (default)
        alpha     = 1.8, # relaxation parameter: 1.8 (default)
        rho_x     = 1e-3, # x equality constraint scaling: 1e-3 (default)
        cg_rate   = 2, # for indirect, tolerance goes down like (1/iter)^cg_rate: 2 (default)
        verbose   = 1, # boolean, write out progress: 1 (default)
        normalize = 0, # boolean, heuristic data rescaling: 1 (default)
        scale     = 3 # if normalized, rescales by this factor: 5 (default)
        ))

    @variable(model,k2[i=1:T,j=1:nparams], start=k[i,j])
    # for i in eachindex(k)
    #     setvalue(k2[i], k[i])
    # end
    @variable(model, res_norm_const[1:T])
    @variable(model, sum_res_norm_const)
    @variable(model, reg_norm_const[1:T-1])
    @variable(model, sum_reg_norm_const)
    # @show size(k)
    loss = 0
    @constraints(model, begin
        res_const[i=1:T], res_norm_const[i] >= norm((y[((i-1)*n+1):(((i-1)*n+1)+n-1),:] -   A[((i-1)*n+1):(((i-1)*n+1)+n-1),:]*k2[i,1:nparams]))
    end)


    dk = diff_fun(k2)


    @constraint(model, reg_const[t=1:T-1], reg_norm_const[t] >= norm(dk[t,1:nparams]))

    @constraint(model, sum_res_norm_const >= sum(res_norm_const))
    @constraint(model, sum_reg_norm_const >= lambda*sum(reg_norm_const))

    @objective(model,Min, sum_reg_norm_const + sum_res_norm_const)

    status = solve(model)

    At,Bt = ABfromk(getvalue(k2),n,m,T)
    SimpleLTVModel{eltype(At)}(At,Bt,extend)
end
