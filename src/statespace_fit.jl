export fit_statespace, fit_statespace_gd, fit_statespace_constrained, fit_statespace_gd!
using ProximalOperators, SparseArrays, LinearAlgebra, DiffResults


function fit_statespace_gd(d::AbstractIdData,λ; initializer::Symbol=:kalman, extend=false,kwargs...)
    @assert hasinput(d) "The identification data must have inputs for this method to apply"
    model = SimpleLTVModel(d, extend=extend)
    y,A     = matrices(model,d)
    T = length(d)
    n = nstates(d)
    m = ninputs(d)
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
        model = KalmanModel(d,R1,R2,P0, extend=false)
    end
    fit_statespace_gd!(model,d,λ; extend=extend, kwargs...)
end

function fit_statespace_gd!(model::AbstractModel,d::AbstractIdData,λ; normType = 1, D = 1, step=0.001, iters=10000, lasso=0, decay_rate=0.999, momentum=0.9, print_period = 100, reduction=0, extend=true, opt=LBFGS(m=100), kwargs...)
    if reduction > 0
        decay_rate = decayfun(iters, reduction)
    end
    T = length(d)
    n = nstates(d)
    m = ninputs(d)
    x = state(d)
    k = model2statevec(model)
    y, A     = matrices(model,d)
    nparams = size(A,2)
    T      -= 1
    diff_fun = D == 2 ? x-> diff(diff(x,dims=1),dims=1) : x-> diff(x,dims=1)
    function lossfun(k2)
        loss    = 0.
        for i = 1:T
            ii = (i-1)*n+1
            ii2 = ii+n-1
            @inbounds @views loss += mean((y[ii:ii2,:] - A[ii:ii2,:]*k2[i,:]).^2)
        end
        NK = length(k2)
        if normType == 1
            loss += λ^2/NK*sum(sqrt,sum(diff_fun(k2./std(k2,dims=1)).^2, dims=2) )
        else
            loss += λ/NK*sum(diff_fun(k2).^2)
        end
        if lasso > 0
            loss += lasso^2*sum(abs, k2)
        end
        loss
    end


    # inputs      = (k,)
    # loss_tape   = GradientTape(lossfun, inputs)
    # results     = similar.(inputs)
    # all_results = DiffResults.GradientResult.(results)
    #
    # mom         = zeros(size(k))
    # @show bestcost    = lossfun(k)
    # costs       = Inf*ones(iters+1); costs[1] = bestcost
    # steps       = zeros(iters)
    # # opt         = RMSpropOptimizer(k, step, 0.8, momentum)
    # opt         = ADAMOptimizer(k; α = step,  β1 = 0.9, β2 = 0.999, ɛ = 1e-8)
    # for iter = 1:iters
    #     steps[iter] = step
    #     gradient!(all_results, loss_tape, inputs)
    #     costs[iter] = all_results[1].value
    #     # mom .= step.*all_results[1].derivs[1] .+ momentum.*mom
    #     # k .-= mom
    #     opt(all_results[1].derivs[1], iter)
    #     # opt(all_results.derivs, iter)
    #     if iter % print_period == 0
    #         println("Iteration: ", iter, " cost: ", round(costs[iter],digits=6), " stepsize: ", step)
    #     end
    #     step *=  decay_rate
    #     opt.α = step
    # end


    inputs      = k
    loss_tape   = GradientTape(lossfun, inputs)
    results     = similar(inputs)
    all_results = DiffResults.GradientResult(results)
    gradfun = (G,inputs) -> gradient!(G, loss_tape, inputs)
    costs = Optim.optimize(lossfun, gradfun, k, opt, Optim.Options(store_trace=true, show_trace=true, show_every=10, iterations=10000, allow_f_increases=false, time_limit=100, x_tol=0, f_tol=0, g_tol=1e-8, f_calls_limit=iters, g_calls_limit=0))

    k = costs.minimizer

    At,Bt = ABfromk(k,n,m,T)
    SimpleLTVModel(At,Bt,extend), costs
end



function fit_statespace_constrained(d::AbstractIdData,changepoints::AbstractVector; extend=true)
    T             = length(d)
    n             = nstates(d)
    m             = ninputs(d)
    y,A           = matrices(SimpleLTVModel(d), d)
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

"""
model = fit_admm(modeltype::Type, d::AbstractIdData, λ; initializer::Symbol=:kalman)

Fit a model by solving `minimize ||y-ŷ||² + λ²||Dₓ k||` where `x` is 1 or 2
using linearized ADMM

See README for usage example

# Keyword arguments
iters      = 10000
D          = 1 # Order of differentiation
tol        = 1e-5
printerval = 100
zeroinit   = false
cb         = nothing # Callback function `model -> cb(model)`
γ          = 0.05 # Regularization parameter
μ          = γ/4/D^2/(ridge == 0 ? 1 : 2), # 32 is the biggest possible ||A||₂² # ADMM parameter
ridge      = 0 # `ridge > 0` Add some L2 regularization (`||k||`)
"""
function fit_admm(model::AbstractModel, d::AbstractIdData,λ; initializer::Symbol=:kalman, zeroinit = false, kwargs...)
    y,A  = matrices(model, d)
    T = length(d)
    n = nstates(d)
    m = ninputs(d)
    FT = eltype(y)
    T -= 1
    if !zeroinit
        if initializer != :kalman
            k = A\y
            k = repmat(k',T,1)
            model = statevec2model(model,k,n,m,false)
        else
            R1 = FT.(0.1*eye(n^2+n*m))
            R2 = FT.(10eye(n))
            P0 = 10000R1
            model = KalmanModel(model,d,R1,R2,P0, extend=false)
        end
    end
    fit_admm!(model, d,λ; zeroinit=zeroinit, kwargs...)
end

function fit_admm!(model::AbstractModel,d::AbstractIdData,λ;
    iters      = 10000,
    D          = 1,
    tol        = Float64(1e-5),
    printerval = 100,
    zeroinit   = false,
    cb         = nothing,
    γ          = Float64(0.05),
    ridge      = 0,
    μ          = Float64(γ/4/D^2/(ridge == 0 ? 1 : 2)), # 32 is the biggest possible ||A||₂²
    kwargs...)

    T = length(d)
    T -= 1
    n = noutputs(d)
    m = ninputs(d)
    γ, ridge
    k       = LTVModels.model2statevec(model) |> copy
    y, Φ    = matrices(model,d)
    nparams = size(Φ,2)
    @show n, size(y), size(Φ)
    y       = reshape(y,n,:)
    FT      = eltype(y)
    NK      = length(k)
    x       = !zeroinit*copy(k[:])
    A       = sparse(FT(1.0)*I,NK,NK)
    if D == 1
        normA2 = ridge > 0 ? 2*3.91 : 3.91
        z       = !zeroinit*diff(k,dims=2)[:]
        for i = 1:NK-nparams
            A[i, i+nparams] = -1
        end
        A       = A[1:end-nparams,:]
    elseif D == 2
        normA2 = ridge > 0 ? 2*15.1 : 15.1
        z       = !zeroinit*diff(diff(k,dims=2),dims=2)[:]
        for i = 1:NK-nparams
            A[i, i+nparams] = -2
        end
        for i = 1:NK-2nparams
            A[i, i+2nparams] = 1
        end
        A       = A[1:end-2nparams,:]
    end
    @assert 0 ≤ μ ≤ γ/normA2 "μ should be ≤ γ/$normA2"

    proxf = prox_ls(model, y, Φ)

    gs = ntuple(t->NormL2(λ), T-D)
    indsg = ntuple(t->((t-1)*nparams+1:t*nparams, ) ,T-D)

    ## To add extra penalty
    if ridge > 0
        Q     = sparse(FT(ridge)*I(nparams))
        q     = zeros(FT,nparams)
        gs2   = fill(Quadratic(Q,q), T-D)
        gs    = (gs...,gs2...)
        indsg = (indsg..., indsg...)
        A     = [A; A]
        z     = [z;z]
    end
    ##

    proxg = SlicedSeparableSum(gs, indsg)
    @show size(A), size(x), size(y), size(Φ)
    Ax        = A*x
    u         = zeros(FT,size(z))
    Axz       = Ax .- z
    Axzu      = similar(u)
    proxf_arg = similar(x)
    proxg_arg = similar(u)
    x,z = _fit_admm_inner(Axzu, Axz,u,μ,γ,A,x,proxf_arg,proxf,z,Ax,proxg,proxg_arg,n,m,T,tol,cb,printerval,iters)
    k = reshape(x,nparams,T)' |> copy
    model = LTVModels.statevec2model(typeof(model),k,n,m,true)
    # plot(flatten(model))
    model
end



function _fit_admm_inner(Axzu, Axz,u,μ,γ,A,x,proxf_arg,proxf,z,Ax,proxg,proxg_arg,n,m,T,tol,cb,printerval,iters)
    for i = 1:iters

        Axzu .= Axz.+u
        proxf_arg .= -(μ/γ) .* A'Axzu .+ x # proxf_arg .= x - (μ/γ)*A'*Axzu
        # proxf_arg .+= x

        prox!(x, proxf, proxf_arg, μ)
        Ax .= A*x # Ax       .= A*x
        proxg_arg .= Ax .+ u
        prox!(z, proxg, proxg_arg, γ)
        Axz .= Ax .- z
        u  .+= Axz

        nAxz = norm(Axz)
        if i % printerval == 0
            @printf("%d ||Ax-z||₂ %.6f\n", i,  nAxz)
            if cb !== nothing
                cb(reshape(x,:,T) |> copy)
            end
        end
        if nAxz < tol
            @info("||Ax-z||₂ ≤ tol")
            break
        end
    end
    x,z
end




function prox_ls(model::SimpleLTVModel, y, Φ)
    nparams = size(Φ,2)
    n,T = size(y)
    fs = ntuple(T) do t
        ii  = (t-1)*n+1
        ii2 = ii+n-1
        a   = Φ[ii:ii2,:]
        Q   = Matrix(a'a)
        q   = -a'y[:,t]
        ProximalOperators.QuadraticIterative(2Q,2q)
    end
    indsf = ntuple(t->((t-1)*nparams+1:t*nparams, ), T)
    proxf = SlicedSeparableSum(fs, indsf)
end

# function prox_ls(model::LTVAutoRegressive, y, Φ)
#     LeastSquares(Φ,vec(y)) # This prox is not correct
# end


struct ARProx3{AT,YT} <: ProximableFunction
    ATA::AT
    Ay::YT
    na::Int
end

function ProximalOperators.prox!(o, f::ARProx3, x, γ)
    # TODO: this can be done vastly more efficiently
    ATA,Ay = f.ATA, f.Ay
    # s = 0.
    na = f.na
    # @show size(A), size(o)
    o = reshape(o, na,:)
    x = reshape(x, na,:)
    # o = reshape(o, :, na)
    # x = reshape(x, :, na)
    for i in 1:size(x,1)
        @views o[:,i] .= (Ay[i] + 1/γ*x[:,i])./(ATA[i] + 1/γ)
        # s +-
    end
    # s
end

function prox_ls(model::LTVAutoRegressive, y, Φ)
    nparams = size(Φ,2)
    T = length(y)
    n = model.na
    # T -= 1
    ATA = [a'a for a in eachrow(Φ)]
    Ay = [Φ[i,:].*y[i] for i in 1:T]
    ARProx3(ATA,Ay,n)
end
