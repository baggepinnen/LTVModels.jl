using GaussianMixtures
using Distributions
using PDMats
import Base.LinAlg.Cholesky

type SingularGMM
    K::Int
    D::Int
    μ::Matrix{Float64}
    Σ::Vector{Matrix{Float64}}
    w::Vector{Float64}
    SingularGMM(K,D) = new(K,D,zeros(K,D),Matrix{Float64}[zeros(D,D) for i=1:K], zeros(K))
end

type LinearDynamics
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    fx::Matrix{Float64}
    fu::Matrix{Float64}
end
LinearDynamics(d::Int) = LinearDynamics(zeros(d),zeros(d,d),zeros(d,d),zeros(d,d))

type AffineTransform
    U::Matrix{Float64}
    xmin::Vector{Float64}
    xmax::Vector{Float64}
end


# """
# Transforms a ´point´ to the ´Lower´ dimensional space
# """
function downtransform(T::AffineTransform, p)
    p .-= T.xmin
    p ./= T.xmax
    p   = T.U'*p
end


"""
Transforms a ´point´ to the ´Higher´ dimensional space
"""
function uptransform(T::AffineTransform, p)
    p   = T.U*p
    p .*= T.xmax
    p .+= T.xmin
end

"""
`strip(Tin::AffineTransform, i)`

Keep only dimensions in the index vector `i`
"""
function Base.strip(Tin::AffineTransform, i)
    T      = deepcopy(Tin)
    T.U    = T.U[i,:]
    T.xmax = T.xmax[i]
    T.xmin = T.xmin[i]
    return T
end

"""
`model = fit_good_model(X, K, nTries)`

This function tries to fit a GMM with `K` clusters a number of times and selects the best model
"""
function fit_good_model(X, K, nTries)
    models = Array(GMM,nTries)
    for i = 1:nTries
        itworked = 0
        while  10 > itworked >= 0
            try
                models[i] = GMM(K, X'; method=:kmeans, kind=:full, nInit=50, nIter=100, nFinal=100)
                itworked = -1
            catch ex
                itworked += 1
                println(ex)
            end
        end
        itworked == 10 && error("Better luck another time")
    end
    model = models[indmax([avll(models[i],X') for i = 1:nTries])]
end

"""
Get assignments for the data in `X` under the `model`

`Q,ass = get_assignments(model::GMM, X)`
"""
function get_assignments(model::GMM, X)
    N = size(X,2)
    K = model.n
    Q = zeros(N,K)
    Σ = covars(model)
    for n = 1:N
        for k = 1:K
            #             Γ = PDMat(Cholesky(model.Σ[k].data,:U))
            μ = vec(model.μ[k,:]')
            Q[n,k] = model.w[k]*pdf(MvNormal(μ, Σ[k]),vec(X[:,n]))
        end
        Q[n,:] ./= sum(Q[n,:])
    end
    ass = findmax(Q,2)[2][:]
    ass = [ind2sub(Q,a)[2] for a in ass][:]
    if N >= K
        1:K ⊆ ass || warn("Not all clusters have data points assigned to them")
    end
    return Q,ass
end

# """
# Get assignments for the data in `X` under the `model`
#
# `ass = get_assignments_robust(model::GMM, X)`
# """
function get_assignments_robust(model::GMM, X)
    N = size(X,2)
    K = model.n
    Q = zeros(K)
    Σ = covars(model)
    ass = zeros(Int, N)
    for n = 1:N
        for k = 1:K
            μ = vec(model.μ[k,:]')
            Q[k] = model.w[k]*logpdf(MvNormal(μ, Σ[k]),vec(X[:,n]))
        end
        ass[n] = indmax(Q)
    end
    return ass
end

function get_assignments(model::GMM, X̄, T)
    X = downtransform(T, X̄)
    get_assignments(model::GMM, X)
end

function get_assignments_robust(model::GMM, X̄, T)
    X = downtransform(T, X̄)
    get_assignments_robust(model::GMM, X)
end


function conditionalGMM(M, d, p, T)
    K,D  = M.K, M.D
    r    = rank(M.Σ[1])
    Sf   = 1:D
    Sl   = d
    Sr   = setdiff(Sf,Sl)
    Tr   = deepcopy(T)
    Tr.U = Tr.U[Sr,:]
    Tl   = deepcopy(T)
    Tl.U = Tl.U[Sl,:]
    pr   = downtransform(Tr,p)
    Mc   = GMM(K,length(d))
    Mc.w = M.w
    J    = Vector{Matrix{Float64}}(K)
    Q    = Vector{Float64}(K)
    f    = Vector{Float64}(K)
    for k = 1:K
        Σ          = M.Σ[k]
        μ          = M.μ[k,:]'
        J[k]       = Σ[Sl,Sr]/Σ[Sr,Sr]
        Mc.μ[k,:]  = (μ[Sl] + J[k]*(pr - μ[Sr]))'
        Mc.Σ[k]    =  Σ[Sl,Sl] - J[k]*Σ[Sr,Sl]
        # J[k]     = Tl.U*A*Tr.U'
        warn("Have not yet considered scaling by Xmax")
        f[k] = fN(vec(M.μ[k,Sr]), M.Σ[k][Sr,Sr], pr, regul)[1]
        Q[k] = M.w[k]*f[k]
    end

    Q ./= sum(Q)

    return Mc, J
end


# """
# calculates the model p(x^+ | x, u)
# where x^+ is p[d] and [x;u] is p[!d]
# dof is the numer of degrees of freedom, like 7 for the YuMi robot
# """

function conditional_dynamics(M, d, p, dof, T)
    (K,D) = (M.K, M.D)
    Mc, Jk= conditionalGMM(M, xp_indices, p, T)
    dyn   = LinearDynamics(length(d))

    J   = zeros(length(Sl),length(Sr))
    for k = 1:K
        dyn.μ += Q[k]*Mc.μ[k,:]
        dyn.Σ += Q[k]*Mc.Σ[k]
        J     += Q[k]*Jk[k]
    end

    dyn.fx   = J[:,1:dof]
    dyn.fu   = J[:,dof+1:end]
    return dyn
end


function normalize_data(x, tol=0.1)
    mi = mean(x,1)
    x .-= mi
    ma = (maximum(x,1)+tol)
    x ./= ma
    return x, mi, ma
end

"""
`Λ(A,r::Int)` This function takes an input matrix A and regularizes it so that all singular values σi=σr ∀ i ≧ r = rank(A)
The idea is that singular covariance matrices can be regularized so that they can be inverted. This has the effect of adding some small amount of uncertainty (variance) in the directions that have zero (very small) variance
"""
function Λsvd(A,r::Int)
    (U,S,V) = svd(A)
    if r < length(S)
        S[r+1:end] = S[r]
    end
    return U*diagm(S)*V'
end

"""
`Λ(A,r::Int)` This function takes an input matrix A and regularizes it so that all eigen values λi=λr ∀ i ≧ r = rank(A)
The idea is that singular covariance matrices can be regularized so that they can be inverted. This has the effect of adding some small amount of uncertainty (variance) in the directions that have zero (very small) variance
"""
function Λ(A,r::Int)
    e = eigfact((A+A')/2)
    if r < length(e.values)
        e.values[1:end-r] = e.values[end-r+1]
    end
    return e.vectors*diagm(e.values)*e.vectors'
end

"""
`Λ(A,r::Float64=1e-5)` This function takes an input matrix A and regularizes it by adding a small multiple of the identity matrix
The idea is that singular covariance matrices can be regularized so that they can be inverted. This has the effect of adding some small amount of uncertainty (variance) in the directions that have zero (very small) variance
"""
function Λ(A,r::Float64=1e-5)
    return A + r*I
end


"""
fN(M::MvNormal, x) -> (f,df) calculates the pdf and the derivative of the pdf of the MvNormal M at point x
"""
function fN(μ, Σ, x, r=10_000)
    k   = size(x,1)
    d   = x-μ
    Σid = Σ\d
    e   = exp(-0.5*d'*Σid)[1]
    f   = sqrt((2*pi)^k*det(Σ))*e
    df  = 2Σid*f
    return f,df
end


"""
`dyn = fitDynamics_decoupled(y, x, u; λ = 1e-7)`

This function assumes that there are an equal number of states and control signals, and that the modeling can be decoupled such that each state is affected by a single input.
"""
function fit_dynamics_decoupled(y, x, u; λ = 1e-7)
    N,d = size(x)
    w   = zeros(2d,d)
    n   = d + size(u,2)
    for i = 1:d
        A     = [x[:,i] u[:,i]]
        G     = [A; λ*eye(n)]
        w[[i,i+d],i]     = G\[y[:,i]; zeros(n)]
    end
    ŷ  = A*w
    e  = y-ŷ
    # σ2 = var(e)
    Σ  = cov(e)
    fx = w[1:d,:]
    fu = w[d+1:2d,:]
    #     μ  = w[2d+1:end,:]
    return LinearDynamics(vec(mean(ŷ,1)), Σ, fx, fu)
end

# """
# `dyn = fitDynamics(y, x, u; λ = 1e-7)`
#
# """
function fit_dynamics(y, x, u; λ = 1e-7)
    N,d  = size(x)
    A    = [x u]
    n    = size(A,2)
    G    = [A; λ*eye(n)]
    w    = G\[y; zeros(n,d)]
    ŷ    = A*w
    e    = y-ŷ
    # σ2 = var(e)
    Σ    = cov(e)
    fx   = w[1:d,:]'
    fu   = w[d+1:end,:]'
    # μ  = w[2d+1:end,:]

    return LinearDynamics(vec(mean(ŷ,1)), Σ, fx, fu)
end
