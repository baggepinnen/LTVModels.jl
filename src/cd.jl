# This code is experimental and is likely not working properly
signfunc(x) = x <= 0 ? -1 : 1
S(β, γ) = γ >= abs(β) ? 0. : β > 0 ? β-γ : β+γ
@inline function derivative(t, βt, beta, y, λ)
    retval = -(y[t]-βt)
    if t != length(beta)
        retval -= λ*signfunc(beta[t+1]-βt)
    end
    if t != 1
        retval += λ*signfunc(βt-beta[t-1])
    end
    retval
end
@inline function derivativeγ(t, γ, beta, y, λ)
    retval = -(y[t-1]-γ) - (y[t]-γ)
    if t != length(beta)
        retval -= λ*signfunc(beta[t+1]-γ)
    end
    if t > 2
        retval += λ*signfunc(γ-beta[t-2])
    end
    retval
end

signchange(t, β, beta, y, λ) = derivative(t, β-eps(), beta, y, λ)*derivative(t, β+eps(), beta, y, λ) < 0

@inline function costfun(t, βt, beta, y, λ)
    retval = (y[t]-βt)^2
    if t != length(beta)
        retval += λ*abs(beta[t+1]-βt)
    end
    if t != 1
        retval += λ*abs(beta[t-1]-βt)
    end
    retval
end

@inline @inbounds function switch(p,i1,i2)
    temp = p[i1]
    p[i1] = p[i2]
    p[i2] = temp
end

fastsort!(points::Number) = points
# @inline fastsort!(points) = fastsort2!(points)

@inbounds function fastsort3!(points) # Tested
    if points[3] < points[2]
        if points[3] < points[1]
            switch(points,1,3)
            if points[3] < points[2]
                switch(points,2,3)
            end
            return points
        else
            switch(points,2,3)
            return points # Done
        end
    end
    if points[2] < points[1]
        switch(points,1,2)
    end
    if points[3] < points[2]
        switch(points,2,3)
    end
    return points
end

@inbounds function fastsort!(points)
    if points[2] < points[1]
        switch(points,2,1)
    end
    return points
end

function findsmallest(t, y, beta, λ)
    t == 1 && (return beta[t+1])
    t == length(beta) && (return beta[t-1])
    return costfun(t, beta[t-1], beta, y, λ) < costfun(t, beta[t+1], beta, y, λ) ? beta[t-1] : beta[t+1]
end
function updateβ(t, points, derfun, y, beta, λ)
    fastsort!(points)
    d = derfun(t, points[1], beta, y, λ)
    d > 0 && (return updateβ(t, [points[1]-1, points[1]], derfun, y, beta, λ))
    abs(d) < 1e-10 || signchange(t, points[1], beta, y, λ) && (return points[1])

    d = derfun(t, points[2], beta, y, λ)
    if abs(d) < 1e-10 || signchange(t, points[2], beta, y, λ) # Minimum is at discontinuity
        return points[2]
    end
    d = derfun(t, points[2]-eps(), beta, y, λ)
    if d > 0 # We have passed the minimum and interpolate to find zero
        width = points[2]-points[1]
        dlast = derfun(t, points[1]+eps(), beta, y, λ)
        height = d-dlast
        return points[1] - dlast*width/height # dlast is negative
    end

    return updateβ(t, [points[2]+1, points[2]], derfun, y, beta, λ)
end

function fusedlasso_cd(y,λ; iters=20, λ_increase_factor = 2, kwargs...)
    T           = size(y,1)
    beta        = copy(y)
    costs       = zeros(y)#fill(Inf,size(y))
    lastcosts   = zeros(y)#fill(Inf,size(y))
    costhistory = zeros(iters)
    λvec = logspace(λ/100,λ,10)
    # for (outer_iter, λ) = enumerate(λvec)
    for i = 1:iters

        # Descent cycle ========================================================
        beta[1] = updateβ(1, [beta[1], beta[2]], derivative, y, beta, λ)
        costs[1] = costfun(1, beta[1], beta, y, λ)
        for t = 2:T-1 # Coordinate descent for each parameter
            # if beta[t] == beta[t-1] || beta[t] == beta[t+1]
            #     # do something else
            #     continue
            # end
            beta[t] = updateβ(t, [beta[t-1], beta[t+1]], derivative, y, beta, λ)
            costs[t] = costfun(t, beta[t], beta, y, λ)
        end
        beta[T] = updateβ(1, [beta[T-1], beta[T]], derivative, y, beta, λ)
        costs[T] = costfun(T, beta[T], beta, y, λ)

        # Fusion cycle =========================================================
        for t = 3:T-1 # TODO: fix number 3
            if costs[t] == lastcosts[t]
                # println("Fusing t=", t)
                beta[t] = beta[t-1] = updateβ(t, [beta[t-1], beta[t+1]], derivativeγ, y, beta, λ)
            end
        end
        # Smoothing cycle ======================================================
        # if any(costs .>= lastcosts)
        #     continue # The smoothing cycle
        # end
        copy!(lastcosts, costs)
        costhistory[i] = sum(costs)
    end
    # costhistory[outer_iter] = sum(costs)
    # end
    return beta,costhistory
end

function test_fusedlasso_cd(N; λ = 1, iters = 100)
    y = [-10+randn(N); 10+randn(N)]

    @time yh, cost = fusedlasso_cd(y,λ, iters = iters)
    plot([y yh], layout=2, subplot=1)
    plot!(cost, subplot=2, yscale=:log10, xscale=:log10)
end


function test_derivative()
    y = randn(3)
    beta = copy(y)
    λ = 1
    t = 2
    betagrid = linspace(-3,3,2000)
    derfun = derivative
    derivs = [derfun(t, βt, beta, y, λ) for βt in betagrid]
    costs = [costfun(t, βt, beta, y, λ) for βt in betagrid]
    points = [beta[1], beta[3]]
    newbeta = updateβ(t, points, derfun, y, beta, λ)
    d = derfun(t, newbeta, beta, y, λ)
    @assert abs(d) < 1e-9  || derfun(t, newbeta-1e-10, beta, y, λ)*derfun(t, newbeta+1e-10, beta, y, λ) < 0

    plot(betagrid,[derivs costs])
    plot!(points'.*ones(2), [-5, 5])
    plot!(newbeta*ones(2), [-5, 5], l=:dash)
end
