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
    for jj = M-1:-1:1
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



function fit_statespace_dp(x,u, M; kwargs...)
    T,n = size(x)
    m = size(u,2)
    input = matrices(x,u)
    V,bps,a = seg_bellman(input,M, ones(T-1), cost_ss, argmin_ss, doplot=false)
    At,Bt = segments2full(a,bps,n,m,T)

    LTVModel(At, Bt)
end
