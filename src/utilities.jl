export toeplitz, toOrthoNormal, flatten, segment, segmentplot, rms, modelfit

rms(x::AbstractVector) = sqrt(mean(x.^2))
sse(x::AbstractVector) = x⋅x

rms(x::AbstractMatrix) = sqrt.(mean(x.^2,2))[:]
sse(x::AbstractMatrix) = sum(x.^2,2)[:]
modelfit(y,yh) = 100 * (1-rms(y.-yh)./rms(y.-mean(y)))
aic(x::AbstractVector,d) = log(sse(x)) + 2d/size(x,2)

function toeplitz{T}(c::Array{T},r::Array{T})
    nc = length(c)
    nr = length(r)
    A = zeros(T, nc, nr)
    A[:,1] = c
    A[1,:] = r
    @views for i in 2:nr
        A[2:end,i] = A[1:end-1,i-1]
    end
    A
end

function toOrthoNormal(Ti)
    local T = deepcopy(Ti)
    U_,S_,V_ = svd(T[1:3,1:3])
    local R = U_*diagm([1,1,sign(det(U_*V_'))])*V_'
    T[1:3,1:3] = R
    return T
end

function getD(D,T)
    if D == 3
        return sparse(toeplitz([-1; zeros(T-4)],[-1 3 -3 1 zeros(1,T-4)]))
    elseif D == 2
        return sparse(toeplitz([1; zeros(T-3)],[1 -2 1 zeros(1,T-3)]))
    elseif D == 1
        return sparse(toeplitz([-1; zeros(T-2)],[-1 1 zeros(1,T-2)]))
    end
    error("Can not handle your choice of D: $D")
end

function matrices(x,u)
    n,T = size(x)
    T -= 1
    m = size(u,1)
    A = spzeros(T*n, n^2+n*m)
    y = zeros(T*n)
    I = speye(n)
    for i = 1:T
        ii = (i-1)*n+1
        ii2 = ii+n-1
        A[ii:ii2,1:n^2] = kron(I,x[:,i]')
        A[ii:ii2,n^2+1:end] = kron(I,u[:,i]')
        y[ii:ii2] = (x[:,i+1])
    end
    y,A
end
flatten(A) = reshape(A,prod(size(A,1,2)),size(A,3))'
flatten(model::LTVStateSpaceModel) = [flatten(model.At) flatten(model.Bt)]
decayfun(iters, reduction) = reduction^(1/iters)

function ABfromk(k,n,m,T)
    At = reshape(k[:,1:n^2]',n,n,T)
    At = permutedims(At, [2,1,3])
    Bt = reshape(k[:,n^2+1:end]',m,n,T)
    Bt = permutedims(Bt, [2,1,3])
    At,Bt
end

"""
At,Bt = segments2full(parameters,breakpoints,n,m,T)
"""
function segments2full(parameters,breakpoints,n,m,T)
    At,Bt = zeros(n,n,T), zeros(n,m,T)
    i = 1
    for t = 1:T
        i ∈ breakpoints && (i+=1)
        At[:,:,t] = reshape(parameters[i][1:n^2],n,n)'
        Bt[:,:,t] = reshape(parameters[i][n^2+1:end],n,m)'
    end
    At,Bt
end




segment(res) = segment(res...)
segment(model::LTVStateSpaceModel) = segment(model.At,model.Bt)
function segment(At,Bt, args...)
    diffparams = (diff([flatten(At) flatten(Bt)],1)).^2
    # diffparams .-= minimum(diffparams,1)
    # diffparams ./= maximum(diffparams,1)
    activation = sqrt.(sum(diffparams,2)[:])
    activation
end

function segmentplot(activation, state; filterlength=10, doplot=false, kwargs...)
    plot(activation, lab="Activation")
    ds = diff(state)
    ma = mean(activation)
    ds[ds.==0] = NaN
    segments = findpeaks(activation; filterlength=filterlength, doplot=doplot, kwargs...)
    doplot || scatter!(segments,[3ma], lab="Automatic segments", m=(10,:xcross))
    scatter!(2ma*ds, lab="Manual segments", xlabel="Time index", m=(10,:cross))
end

# filterlength=2; minh=5; threshold=-Inf; minw=18; maxw=Inf; doplot=true
# plot(activationf)
# scatter!(peaks,activationf[peaks])


"""
A,B,x,xnew,u,n,m,N = testdata(T=10000, σ_state_drift=0.001, σ_param_drift=0.001)
"""
function testdata(;T=10000, σ_state_drift=0.001, σ_param_drift=0.001)
    srand(1)
    n           = 3
    m           = 2
    T           = 10000
    N           = n*(n+m)
    A           = zeros(n,n,T)
    B           = zeros(n,m,T)
    x           = zeros(n,T)
    xnew        = zeros(n,T)
    u           = randn(m,T)
    U,S,V       = toOrthoNormal(randn(n,n)), diagm(0.4rand(n)), toOrthoNormal(randn(n,n))
    A[:,:,1]    = U*S*V'
    B[:,:,1]    = 0.5randn(n,m)
    x[:,1]      = 0.1randn(n)

    for t = 1:T-1
        x[:,t+1]   = A[:,:,t]*x[:,t] + B[:,:,t]*u[:,t] + σ_state_drift*randn(n)
        xnew[:,t]  = x[:,t+1]
        A[:,:,t+1] = A[:,:,t] + σ_param_drift*randn(n,n)
        B[:,:,t+1] = B[:,:,t] + σ_param_drift*randn(n,m)
    end
    A,B,x,xnew,u,n,m,N
end

"""
x,xm,u,n,m = testdata(T_)
"""
function testdata(T_)
    srand(1)

    n,m      = 2,1
    At_      = [0.95 0.1; 0 0.95]
    Bt_      = reshape([0.2; 1],2,1)
    u        = randn(1,T_)
    x        = zeros(n,T_)
    for t = 1:T_-1
        if t == T_÷2
            At_ = [0.5 0.05; 0 0.5]
        end
        x[:,t+1] = At_*x[:,t] + Bt_*u[:,t] + 0.2randn(n)
    end
    xm = x + 0.2randn(size(x));
    x,xm,u,n,m
end



# Optimizers ===================================================================


struct RMSpropOptimizer{VecType}
    α::Float64
    rmspropfactor::Float64
    momentum::Float64
    Θ::VecType
    dΘs::VecType
    dΘs2::VecType
end

function RMSpropOptimizer(Θ, α, rmspropfactor=0.8, momentum=0.1)
    RMSpropOptimizer(α, rmspropfactor, momentum, Θ, ones(Θ), zeros(Θ))
end


function apply_gradient!(opt, dΘ)
    opt.dΘs .= opt.rmspropfactor.*opt.dΘs .+ (1-opt.rmspropfactor).*dΘ.^2
    ΔΘ = -opt.α * dΘ
    ΔΘ ./= sqrt.(opt.dΘs.+1e-10) # RMSprop

    # ΔΘ = dΘ./sqrt(dΘs+1e-10).*sqrt(dΘs2) # ADAdelta
    # dΘs2 .= 0.9dΘs2 + 0.1ΔΘ.^2 # ADAdelta
    opt.dΘs2 .= opt.momentum.*opt.dΘs2 .+ ΔΘ # Momentum + RMSProp
    opt.Θ .+= opt.dΘs2
end

(opt::RMSpropOptimizer)(dΘ) = apply_gradient!(opt, dΘ)



mutable struct ADAMOptimizer{T, VecType <: AbstractArray}
    Θ::VecType
    α::T
    β1::T
    β2::T
    ɛ::T
    m::VecType
    v::VecType
end

ADAMOptimizer{T,VecType <: AbstractArray}(Θ::VecType; α::T = 0.005,  β1::T = 0.9, β2::T = 0.999, ɛ::T = 1e-8, m=zeros(Θ), v=zeros(Θ)) = ADAMOptimizer{T,VecType}(Θ, α,  β1, β2, ɛ, m, v)

"""
    (a::ADAMOptimizer{T,VecType})(g::VecType, t::Integer)
Applies the gradient `g` to the parameters `a.Θ` (mutating) at iteration `t`
ADAM GD update http://sebastianruder.com/optimizing-gradient-descent/index.html#adam
"""
function (a::ADAMOptimizer)(g, t::Integer)
    mul = (1-a.β1)
    mul2 = (1-a.β2)
    div  = 1/(1 - a.β1 ^ t)
    div2 = 1/(1 - a.β2 ^ t)
    α,β1,β2,ɛ,m,v,Θ = a.α,a.β1,a.β2,a.ɛ,a.m,a.v,a.Θ
    Base.Threads.@threads for i = 1:length(g)
        @inbounds m[i] = β1 * m[i] + mul * g[i]
        @inbounds v[i] = β2 * v[i] + mul2 * g[i]^2
        @inbounds Θ[i] -= α * m[i] * div / (sqrt(v[i] * div2) + ɛ)
    end
    Θ
end
# function (a::ADAMOptimizer)(g, t::Integer)
#     a.m .= a.β1 .* a.m .+ (1-a.β1) .* g
#     a.v .= a.β2 .* a.v .+ (1-a.β2) .* g.^2
#     div  = 1/(1 - a.β1 ^ t)
#     div2 = 1/(1 - a.β2 ^ t)
#     @. a.Θ  -= a.α * a.m * div / (sqrt(a.v * div2) + a.ɛ)
# end

if false
using Revise, BenchmarkTools, LTVModels
const g = 0.000001randn(1_000_000);
const Θ = zeros(g)
const opt = LTVModels.ADAMOptimizer(Θ)
function f(n)
    for i = 1:n
        opt(g,1000)
    end
end

@time f(1000)
end


@views model2statevec(model,t) = [model.At[:,:,t]  model.Bt[:,:,t]]' |> vec

function model2statevec(model)
    k = [flatten(model.At) flatten(model.Bt)]
    if model.extended
        k = k[1:end-1,:]
    end
    k
end


function statevec2model(k,n,m,extend)
    if extend
        k = [k; k[end,:]]
    end
    T = size(k,1)
    At = zeros(n,n,T)
    Bt = zeros(n,m,T)
    for t = 1:T
        ABt      = reshape(k[:,t],n+m,n)'
        At[:,:,t] .= ABt[:,1:n]
        Bt[:,:,t] .= ABt[:,n+1:end]
    end
    SimpleLTVModel(At,Bt,extend)
end
