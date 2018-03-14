export fit_statespace, fit_statespace_gd, fit_statespace_constrained, fit_statespace_gd!, fit_statespace_jump!
using ProximalOperators
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







#=
using JuMP, SCS, FirstOrderSolvers
function fit_statespace_jump!(model::AbstractModel,x,u,lambda; normType = 1, D = 1,  lasso=0,  extend=true, kwargs...)
    k             = model2statevec(model)
    const y, A    = matrices(x,u)
    nparams       = size(A,2)
    n,T           = size(x)
    T            -= 1
    m             = size(u,1)
    NK            = length(k)
    diff_fun      = D == 2 ? x-> diff(diff(x,1),1) : x-> diff(x,1)
    # model = Model(IpoptSolver())
    # model         = Model(solver=SCSSolver(
    #     max_iters = 40000,
    #     eps       = 1e-5, # convergence tolerance: 1e-3 (default)
    #     alpha     = 1.8, # relaxation parameter: 1.8 (default)
    #     rho_x     = 1e-3, # x equality constraint scaling: 1e-3 (default)
    #     cg_rate   = 2, # for indirect, tolerance goes down like (1/iter)^cg_rate: 2 (default)
    #     verbose   = 1, # boolean, write out progress: 1 (default)
    #     normalize = 0, # boolean, heuristic data rescaling: 1 (default)
    #     scale     = 3 # if normalized, rescales by this factor: 5 (default)
    #     ))
    model = Model(solver=DR())

    @variable(model,k2[i=1:T,j=1:nparams], start=k[i,j])
    for i in eachindex(k)
        setvalue(k2[i], k[i])
    end
    @variable(model, res_norm_const[1:T])
    @variable(model, sum_res_norm_const)
    @variable(model, reg_norm_const[1:T-1])
    @variable(model, sum_reg_norm_const)

    @show size(k)
    loss = 0
    @constraints(model, begin
        res_const[i=1:T], res_norm_const[i] >= norm((y[((i-1)*n+1):(((i-1)*n+1)+n-1),:] -   A[((i-1)*n+1):(((i-1)*n+1)+n-1),:]*k2[i,1:nparams]))
    end)
    # loss = 0
    # for i = 1:T
    #     loss = sum((y[((i-1)*n+1):(((i-1)*n+1)+n-1),:] -   A[((i-1)*n+1):(((i-1)*n+1)+n-1),:]*k2[i,1:nparams]).^2)
    # end


    dk = diff_fun(k2)#.^2
    # dk = sum(dk,2)
    # diffs = dk .== 0


    @constraint(model, reg_const[t=1:T-1], reg_norm_const[t] >= norm(dk[t,1:nparams]))

    @constraint(model, sum_res_norm_const >= sum(res_norm_const))
    @constraint(model, sum_reg_norm_const >= lambda*sum(reg_norm_const))

    @objective(model,Min, sum_reg_norm_const + sum_res_norm_const)


    # @objective(model,Min, loss)
    # @constraint(model, sum(diffs) <= 2)

    status = solve(model)

    At,Bt = ABfromk(getvalue(k2),n,m,T)
    SimpleLTVModel{eltype(At)}(At,Bt,extend)
end

=#

function matrices2(x,u)
    n,T = size(x)
    T -= 1
    m = size(u,1)
    y = x[:,2:end]
    A = spzeros(T*n, n^2+n*m)
    I = speye(n)
    for i = 1:T
        ii = (i-1)*n+1
        ii2 = ii+n-1
        A[ii:ii2,1:n^2] = kron(I,x[:,i]')
        A[ii:ii2,n^2+1:end] = kron(I,u[:,i]')
    end
    y,A
end

using ProximalOperators

function fit_statespace_admm(x,u,lambda; initializer::Symbol=:kalman, extend=false, zeroinit = false, kwargs...)
    y,A     = matrices(x,u)
    n,T     = size(x)
    m       = size(u,1)
    T      -= 1
    if zeroinit
        model = SimpleLTVModel(zeros(n,n,T), zeros(n,m,T), false)
    else
        if initializer != :kalman
            k = A\y
            k = repmat(k',T,1)
            model = statevec2model(k,n,m,false)
        else
            R1 = 0.1*eye(n^2+n*m)
            R2 = 10eye(n)
            P0 = 10000R1
            model = fit_model(KalmanModel, x,u,R1,R2,P0, extend=false)
        end
    end
    fit_statespace_admm!(model, x,u,lambda; extend=extend, zeroinit=zeroinit, kwargs...)
end

function fit_statespace_admm!(model::AbstractModel,x,u,lambda;
    iters      = 10000,
    D          = 1,
    extend     = true,
    tol        = 1e-5,
    printerval = 100,
    zeroinit   = false,
    cb         = nothing,
    λ          = 0.05,
    μ          = λ/4/D^2/(ridge == 0 ? 1 : 2), # 32 is the biggest possible ||A||₂²
    ridge      = 0,
    kwargs...)


    k       = LTVModels.model2statevec(model)'
    y, Φ    = LTVModels.matrices2(x,u)
    nparams = size(Φ,2)
    n,T     = size(x)
    T      -= 1
    m       = size(u,1)
    NK      = length(k)
    x       = !zeroinit*copy(k[:])
    A       = speye(NK)
    if D == 1
        normA2 = ridge > 0 ? 2*3.91 : 3.91
        z       = !zeroinit*diff(k,2)[:]
        for i = 1:NK-nparams
            A[i, i+nparams] = -1
        end
        A       = A[1:end-nparams,:]
    elseif D == 2
        normA2 = ridge > 0 ? 2*15.1 : 15.1
        z       = !zeroinit*diff(diff(k,2),2)[:]
        for i = 1:NK-nparams
            A[i, i+nparams] = -2
        end
        for i = 1:NK-2nparams
            A[i, i+2nparams] = 1
        end
        A       = A[1:end-2nparams,:]
    end

    @assert 0 ≤ μ ≤ λ/normA2 "μ should be ≤ λ/$normA2"

    fs = ntuple(T) do t
        ii  = (t-1)*n+1
        ii2 = ii+n-1
        a   = Φ[ii:ii2,:]
        Q   = full(a'a)
        q   = -a'y[:,t]
        ProximalOperators.QuadraticIterative(2Q,2q)
    end
    indsf = ntuple(t->((t-1)*nparams+1:t*nparams, ), T)
    proxf = SlicedSeparableSum(fs, indsf)

    gs = ntuple(t->NormL2(lambda), T-D)
    indsg = ntuple(t->((t-1)*nparams+1:t*nparams, ) ,T-D)

    ## To add extra penalty
    if ridge > 0
        Q     = ridge*speye(nparams)
        q     = ridge*spzeros(nparams)
        gs2   = fill(Quadratic(Q,q), T-D)
        gs    = (gs...,gs2...)
        indsg = (indsg..., indsg...)
        A     = [A; A]
        z     = [z;z]
    end
    ##

    proxg = SlicedSeparableSum(gs, indsg)
    Ax        = A*x
    u         = zeros(size(z))
    Axz       = Ax .- z
    Axzu      = similar(u)
    proxf_arg = similar(x)
    proxg_arg = similar(u)
    for i = 1:iters

        Axzu .= Axz.+u
        At_mul_B!(proxf_arg,A,Axzu) # proxf_arg .= x - (μ/λ)*A'*Axzu
        scale!(proxf_arg, -(μ/λ))
        proxf_arg .+= x
        prox!(x, proxf, proxf_arg, μ)
        A_mul_B!(Ax,A,x) # Ax       .= A*x
        proxg_arg .= Ax .+ u
        prox!(z, proxg, proxg_arg, λ)
        Axz .= Ax .- z
        u  .+= Axz

        nAxz = norm(Axz)
        if i % printerval == 0
            @printf("%d ||Ax-z||₂ %.6f\n", i,  nAxz)
            if cb != nothing
                k = reshape(x,n*(n+m),T)'
                model = LTVModels.statevec2model(k,n,m,true)
                cb(model)
            end
        end
        if nAxz < tol
            info("||Ax-z||₂ ≤ tol")
            break
        end
    end
    k = reshape(x,nparams,T)'
    model = LTVModels.statevec2model(k,n,m,true)
    # plot(flatten(model))
    model
end
