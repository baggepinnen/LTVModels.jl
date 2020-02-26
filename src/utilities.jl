export toOrthoNormal, flatten, activation, segmentplot, rms, modelfit


function toOrthoNormal(Ti)
    local T = deepcopy(Ti)
    U_,S_,V_ = svd(T[1:3,1:3])
    local R = U_*diagm(0=>[1,1,sign(det(U_*V_'))])*V_'
    T[1:3,1:3] = R
    return T
end

const ⟂ = toOrthoNormal

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

function matrices(x::AbstractArray{FT},u) where FT
    n = size(x,1)
    T = size(x,2)
    T -= 1
    m = size(u,1)
    A = spzeros(FT, T*n, n^2+n*m)
    y = zeros(FT, T*n)
    Is = sparse(FT(1.0)*I,n,n)
    for i = 1:T
        ii = (i-1)*n+1
        ii2 = ii+n-1
        A[ii:ii2,1:n^2] = kron(Is,x[:,i]')
        A[ii:ii2,n^2+1:end] = kron(Is,u[:,i]')
        y[ii:ii2] = (x[:,i+1])
    end
    y,A
end
flatten(A) = reshape(A,prod(size(A)[1:2]),size(A,3))'
flatten(model::LTVStateSpaceModel) = [flatten(model.At) flatten(model.Bt)]
decayfun(iters, reduction) = reduction^(1/iters)

function ABfromk(k,n,m,T)
    At = reshape(k[:,1:n^2]',n,n,T)
    At = permutedims(At, [2,1,3])
    Bt = reshape(k[:,n^2+1:end]',m,n,T)
    Bt = permutedims(Bt, [2,1,3])
    At,Bt
end

@views model2statevec(model,t) = [model.At[:,:,t]  model.Bt[:,:,t]]' |> vec

function model2statevec(model)
    k = flatten(model)
    if model.extended
        k = k[1:end-1,:]
    end
    k
end

function statevec2model(k,n,m,extend)
    At,Bt = ABfromk(k,n,m,size(k,1))
    SimpleLTVModel(At,Bt,extend)
end


"""
    At,Bt = segments2full(parameters,breakpoints,n,m,T)

Takes a parameter vecor and breakpoints and returns At and Bt with length T
"""
function segments2full(parameters,breakpoints,n,m,T)
    At,Bt = zeros(n,n,T), zeros(n,m,T)
    i = 1
    for t = 1:T
        t ∈ breakpoints && (i+=1)
        At[:,:,t] = reshape(parameters[i][1:n^2],n,n)'
        Bt[:,:,t] = reshape(parameters[i][n^2+1:end],m,n)'
    end
    At,Bt
end

activation(model::LTVStateSpaceModel; kwargs...) = activation(model.At,model.Bt; kwargs...)
function activation(At,Bt; normalize=false)
    diffparams = (diff([flatten(At) flatten(Bt)],dims=1)).^2
    if normalize
        diffparams .-= minimum(diffparams,1)
        diffparams ./= maximum(diffparams,1)
    end
    activation = sqrt.(sum(diffparams,dims=2)[:])
    activation
end

segmentplot(model::AbstractModel, state;  kwargs...) = segmentplot(activation(model),state;kwargs...)
function segmentplot(act, state; filterlength=10, doplot=false, kwargs...)
    plot(act, lab="Activation")
    ds = diff(state)
    ma = mean(act)
    ds[ds.==0] = NaN
    segments = findpeaks(act; filterlength=filterlength, doplot=doplot, kwargs...)
    doplot || scatter!(segments,3ma*ones(segments), lab="Automatic segments", m=(10,:xcross))
    scatter!(2ma*ds, lab="Manual segments", xlabel="Time index", m=(10,:cross))
    segments
end


"""
    A,B,x,u,n,m,N = testdata(T=10000, σ_state_drift=0.001, σ_param_drift=0.001, σ_control=1)

Create an LTVModel with Brownian-walk A[t] and B[t]
"""
function testdata(;T=10000, σ_state_drift=0.001, σ_param_drift=0.001, seed=1, σ_control=1)
    Random.seed!(seed)
    n           = 3
    m           = 2
    N           = n*(n+m)
    A           = zeros(n,n,T)
    B           = zeros(n,m,T)
    x           = zeros(n,T)
    u           = σ_control*randn(m,T)
    U,S,V       = ⟂(randn(n,n)), diagm(0=>0.4rand(n)), ⟂(randn(n,n))
    A[:,:,1]    = U*S*V'
    B[:,:,1]    = 0.5randn(n,m)
    x[:,1]      = 0.1randn(n)

    for t = 1:T-1
        x[:,t+1]   = A[:,:,t]*x[:,t] + B[:,:,t]*u[:,t] + σ_state_drift*randn(n)
        A[:,:,t+1] = A[:,:,t] + σ_param_drift*randn(n,n)
        B[:,:,t+1] = B[:,:,t] + σ_param_drift*randn(n,m)
    end
    A,B,x,u,n,m,N
end

"""
    x,xm,u,n,m = testdata(T)

Create an LTVModel that changes from `A = [0.95 0.1; 0 0.95]` to `A = [0.5 0.05; 0 0.5]`
at T÷2
"""
function testdata(T_)
    Random.seed!(1)

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
    RMSpropOptimizer(α, rmspropfactor, momentum, Θ, ones(size(Θ)), zeros(size(Θ)))
end

function (opt::RMSpropOptimizer)(dΘ)
    opt.dΘs .= opt.rmspropfactor.*opt.dΘs .+ (1-opt.rmspropfactor).*dΘ.^2
    ΔΘ = -opt.α * dΘ
    ΔΘ ./= sqrt.(opt.dΘs.+1e-10) # RMSprop

    # ΔΘ = dΘ./sqrt(dΘs+1e-10).*sqrt(dΘs2) # ADAdelta
    # dΘs2 .= 0.9dΘs2 + 0.1ΔΘ.^2 # ADAdelta
    opt.dΘs2 .= opt.momentum.*opt.dΘs2 .+ ΔΘ # Momentum + RMSProp
    opt.Θ .+= opt.dΘs2
end


mutable struct ADAMOptimizer{T, VecType <: AbstractArray}
    Θ::VecType
    α::T
    β1::T
    β2::T
    ɛ::T
    m::VecType
    v::VecType
    expdecay::Float64
end

ADAMOptimizer(Θ::VecType; α::T = 0.005,  β1::T = 0.9, β2::T = 0.999, ɛ::T = 1e-8, m=zeros(size(Θ)), v=zeros(size(Θ)), expdecay=0) where {T,VecType <: AbstractArray} = ADAMOptimizer{T,VecType}(Θ, α,  β1, β2, ɛ, m, v, expdecay)

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
    α,β1,β2,ɛ,m,v,Θ,γ = a.α,a.β1,a.β2,a.ɛ,a.m,a.v,a.Θ,a.expdecay
    γ *= t > 5
    Base.Threads.@threads for i = eachindex(g)
        @inbounds m[i] = β1 * m[i] + mul * (g[i] + γ*Θ[i])
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
