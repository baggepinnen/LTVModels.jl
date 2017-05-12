
function fit_statespace(x,u,lambda; normType = 2, D = 2, solver=:ECOS, kwargs...)
    y,A  = matrices(x,u)
    T,n  = size(x)
    T   -= 1
    m    = size(u,2)

    Dx   = getD(D,T)
    k    = Convex.Variable(T,n^2+n*m)

    loss = 0
    for i = 1:T
        ii = (i-1)*n+1
        ii2 = ii+n-1
        loss += Convex.sumsquares((y[ii:ii2,:]) - A[ii:ii2,:]*k[i,:]')
    end
    loss += lambda*Convex.norm(Dx*k,normType)
    loss += Convex.sumsquares(k)
    problem = Convex.minimize(loss)
    if solver == :ECOS
        Convex.solve!(problem, ECOS.ECOSSolver(maxit=100, feastol=1e-10, feastol_inacc=1e-5, reltol=1e-09, abstol=1e-09, nitref=50))
    else
        Convex.solve!(problem, SCS.SCSSolver(max_iters=100000, normalize=0, eps=1e-8))
    end

    At,Bt = ABfromk(k,n,m,T)
    LTVModel(At,Bt)
end





function Lcurve(fun, lambdas)
    errors = pmap(fun, lambdas)
    plot(lambdas, errors, xscale=:log10, m=(:cross,))
    errors, lambdas
end



function fit_statespace_gd(x,u,lambda, initializer::Symbol=:kalman ;kwargs...)
    y,A     = matrices(x,u)
    T,n     = size(x)
    m       = size(u,2)
    T      -= 1
    if initializer != :kalman
        k = A\y
        k = repmat(k',T,1)
        k .+= 0.00001randn(size(k))
    else
        model = fit_model(KalmanModel, x[1:end-1,:],u[1:end-1,:],x[2:end,:],0.00001*eye(n^2+n*m),eye(n), false)
        k = [flatten(model.A) flatten(model.B)]
    end
    fit_statespace_gd(x,u,lambda, k; kwargs...)
end
function fit_statespace_gd(x,u,lambda, k; normType = 1, D = 2, step=0.001, iters=10000, decay_rate=0.999, momentum=0.9, reduction=0, adaptive=true, kwargs...)
    if reduction > 0
        decay_rate = decayfun(iters, reduction)
    end
    y,A     = matrices(x,u)
    nparams = size(A,2)
    T,n     = size(x)
    T      -= 1
    m       = size(u,2)
    bestk   = copy(k)
    diff_fun = D == 2 ? x-> diff(diff(x,1),1) : x-> diff(x,1)
    function lossfun(k2)
        loss    = 0.
        for i = 1:T
            ii = (i-1)*n+1
            ii2 = ii+n-1
            loss += mean((y[ii:ii2,:] - A[ii:ii2,:]*k2[i,:]).^2)
        end
        NK = length(k2)
        if normType == 1
            loss += lambda/NK*sum( sqrt(sum(diff_fun(k2).^2, 2)) )
        else
            loss += lambda/NK*sum(diff_fun(k2).^2)
        end
        loss
    end
    loss_tape   = ReverseDiff.GradientTape(lossfun, (k,))
    inputs      = (k,)
    results     = (similar(k),)
    all_results = map(DiffBase.GradientResult, results)
    mom         = zeros(k)
    bestcost    = lossfun(k)
    costs       = Inf*ones(iters+1); costs[1] = bestcost
    steps       = zeros(iters)
    for iter = 1:iters
        steps[iter] = step
        ReverseDiff.gradient!(all_results, loss_tape, inputs)
        mom .= step.*all_results[1].derivs[1] + momentum*mom
        k .-= mom
        costs[iter+1] = lossfun(k) #all_results[1].value
        iter % 100 == 0 && println("Iteration: ", iter, " cost: ", round(costs[iter],6), " stepsize: ", step)
        if costs[iter+1] <= bestcost
            # bestk .= k
            bestcost = costs[iter+1]
            step *= (adaptive ? 10^(1/200) : decay_rate)
        elseif iter > 10 && costs[iter+1] >= costs[iter-10]
            step /= (adaptive ? 10^(1/200) : decay_rate)
            # k .= 0.5.*bestk .+ 0.5.*k; mom .*= 0
        end
    end

    At,Bt = ABfromk(k,n,m,T)
    LTVModel(At,Bt),costs, steps
end

function fit_statespace_constrained(x,u,changepoints::AbstractVector)
    y,A           = matrices(x,u)
    n             = size(x,2)
    m             = size(u,2)
    nc            = length(changepoints)
    changepointse = [1; changepoints; size(x,1)-1]
    Ai            = zeros(n,n,nc+1)
    Bi            = zeros(n,m,nc+1)
    k = map(1:nc+1) do i
        ii = (changepointse[i]-1)*n+1
        ii2 = changepointse[i+1]*n-1
        inds = ii:ii2
        k = (A[inds,:]\y[inds])'
    end
    At,Bt = segments2full(k,bps,n,m,T)
    return LTVModel(At,Bt)
end
