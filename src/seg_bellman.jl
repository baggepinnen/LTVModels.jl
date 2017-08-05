export fit_statespace_dp

# Piecewise constant segmentation ==============================================
@inline function argmin_const(y,w,t1,t2)
    # y'w/sum(w) weighted mean of y
    s = 0.
    dp = 0.
    for t = t1:t2
        dp += y[t]*w[t]
        s += w[t]
    end
    dp/s
end
@inline function cost_const(y,w,t1,t2,argminfun)
    # w'(y-dpamin)^2 weigted variance of y
    dp = 0.
    dpm = argminfun(y,w,t1,t2)
    for t = t1:t2
        dp += w[t]*(y[t]-dpm)^2
    end
    dp
end


# Piecewise statespace segmentation ============================================
@inline function argmin_ss(input,w,t1,t2)
    y   = input[1]
    A   = input[2]
    n   = size(y,1) ÷ length(w)
    ii  = (t1-1)*n+1
    ii2 = t2*n-1
    A[ii:ii2,:]\y[ii:ii2] # TODO: does not take care of w yet
end
@inline function cost_ss(input,w,t1,t2,argminfun)
    y   = input[1]
    A   = input[2]
    n   = size(y,1) ÷ length(w)
    ii  = (t1-1)*n+1
    ii2 = t2*n-1
    k   = argminfun(input,w,t1,t2)
    e   = y[ii:ii2]-A[ii:ii2,:]*k
    return e⋅e
end

using Base.Threads
function seg_bellman(y,M,w, costfun=cost_const, argminfun=argmin_const; doplot=false)
    n = length(w)
    @assert (M < n) "M must be smaller than the length of the data sequence"
    B = zeros(Int,M-1,n) # back-pointer matrix

    # initialize Bellman iteration
    fi = zeros(n)
    for jj = M:n
        fi[jj] = costfun(y,w,jj,n,argminfun)
    end
    doplot && plot(fi, lab="fi init")
    memorymat = fill(Inf,n,n)

    # iterate Bellman iteration
    fnext = Vector{Float64}(n)
    @progress for jj = M-1:-1:1
        for kk = jj:n-(M-jj)
            opt   = Inf
            optll = 0
            for ll = kk+1:n-(M-jj-1)
                if memorymat[kk,ll-1] == Inf
                    memorymat[kk,ll-1] = costfun(y,w,kk,ll-1,argminfun)
                end
                cost = memorymat[kk,ll-1] + fi[ll]
                if cost < opt
                    opt   = cost
                    optll = ll
                end
            end
            fnext[kk] = opt
            B[jj,kk]  = optll-1
        end
        fi .= fnext
        fnext = Vector{Float64}(n)
        doplot && plot!(fi, lab="fi at $jj")
    end

    # last Bellman iterate
    V = [(memorymat[1,jj] == Inf ? costfun(y,w,1,jj,argminfun) : memorymat[1,jj])+fi[jj+1] for jj = 1:n-M]
    if isempty(V) || length(V) <= M
        error("Failed to perform segmentation, try lowering M")
    end
    doplot && plot!(V, lab="V")

    # backward pass
    t = Vector{Int}(M);
    a = Vector{typeof(argminfun(y,w,1,2))}(M+1);
    _,t[1] = findmin(V[1:end-M])
    a[1] = argminfun(y,w,1,t[1])

    for jj = 2:M
        t[jj] = B[jj-1,t[jj-1]]
        a[jj] = argminfun(y,w,t[jj-1]+1,t[jj])
    end
    a[M+1] = argminfun(y,w,t[M]+1,n)

    V,t,a
end



function fit_statespace_dp(x,u, M; extend=true, kwargs...)
    n,T = size(x)
    m = size(u,1)
    input = matrices(x,u)
    V,bps,a = seg_bellman(input,M, ones(T-1), cost_ss, argmin_ss, doplot=false)
    At,Bt = segments2full(a,bps,n,m,T)

    SimpleLTVModel(At, Bt,extend)
end



# Tests
# Dynamic programming ==========================================================
function benchmark_const(N, M=1, doplot=false)
    n = 3N
    y = [0.1randn(N); 10+0.1randn(N); 20+0.1randn(N)+linspace(1,10,N)]
    V,t,a = @time seg_bellman(y,M, ones(y))
    if doplot
    tplot = [1;t;n];
    aplot = [a;a[end]];
    plot(y)
    plot!(tplot, aplot, l=:step);gui()
end
    # yhat, x, a, b
end

function benchmark_ss(T_, M, doplot=false)

    # M        = 1
    # T_       = 400
    x,xm,u,n,m = testdata(T_)
    input = matrices(xm,u)
    @time V,t,a = seg_bellman(input,M, ones(T_-1), cost_ss, argmin_ss, doplot=false)
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
        plot!([1,T_÷2-1], [0.95 0.1; 0 0.95][:]'.*ones(2), ylims=(-0.1,1), l=(:dash,:black, 1))
        plot!([T_÷2,T_], [0.5 0.05; 0 0.5][:]'.*ones(2), l=(:dash,:black, 1), grid=false)
        gui()
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
# M_vec = [2,4,8,16,32]
# times = map(M_vec) do M
#     println(M)
#     @elapsed benchmark_ss(110, M)
# end
# plot(M_vec, times)
