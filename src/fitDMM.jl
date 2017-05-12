using DirichletProcessMixtures
using Distributions
using ConjugatePriors

import ConjugatePriors.NormalWishart

if false
    function ball(N::Int64, x::Float64, y::Float64)
        return randn(2, N) .+ [x, y]
    end

    function balley(M::Int64, R::Float64)
        return hcat(ball(M, 0., 0.),
                    ball(M,  R,  R),
                    ball(M,  R, -R),
                    ball(M, -R,  R),
                    ball(M, -R, -R))
    end

    B = 60
    X = balley(B, 3.)
    xtest = balley(B, 3.)

    N = B * 5
    M = N

    (D,N) = size(x)

    K = 4
    kappa = 1e-7
    ν = K+0.001
    μ = zeros(D)
    W = eye(D) / K
else
    include("read_log.jl")
    include("/local/home/fredrikb/SystemIdentification/src/PCA.jl")
    pathopen = "/home/martinka/Projects/SARAFun/bp-ar-hmm/dmp/data/dmp_frida.txt"
    pathsave = "/local/home/fredrikb/work/sarafun/DDP/log.mat"
    # orcalog2mat(pathopen, pathsave)
    data = MAT.matread(pathsave)

    ds = 1
    q = getData("robot_0.*posRawAbs",data, ds, removeNaN = false)
    tau = getData("robot_0.*trqRaw",data, ds, removeNaN = false)


    # using PyPlot
    # PyPlot.subplot(211)
    # PyPlot.plot(q)
    # PyPlot.subplot(212)
    # PyPlot.plot(tau)

    x = [q[1:end-1,:] tau[1:end-1,:] ]
    y = q[2:end,:]

    x .-= minimum(x,1)
    x ./= (maximum(x,1)+0.01)
    y .-= minimum(y,1)
    y ./= (maximum(y,1)+0.01)


    C,score,latent,W0 = PCA([x y])
    # PyPlot.subplot(313)
    # PyPlot.plot3D(score[:,1],score[:,2],score[:,3],".")
    # PyPlot.show()
    # plot(cumsum(latent)/sum(latent))

    X = score[:,1:5]'
    Xtest = X
    # X = [x y]
    (D,N) = size(X)
    M = N
    K = 4
    kappa = 1e-2
    ν = 500
    μ = zeros(D)
    W = eye(D) / K


end

prior = NormalWishart(μ, kappa, W, ν)

alpha = 10.0
T = 20
maxiter = 4000
gm, theta, predictive_likelihood = gaussian_mixture(prior, T, alpha, X)

lb_log = zeros(maxiter)
tl_log = zeros(maxiter)

tic()
function iter_callback(mix::TSBPMM, iter::Int64, lower_bound::Float64)
    pl = sum(predictive_likelihood(Xtest)) / M
    lb_log[iter] = lower_bound
    tl_log[iter] = pl
    (iter % 10 == 0) && println("iteration $iter test likelihood=$pl, lower_bound=$lower_bound")
end

niter = infer(gm, maxiter, 1e-12; iter_callback=iter_callback)
ass = map_assignments(gm)
@show sass = Set(ass)

using PyPlot
#convergence plot
figure()
subplot(211); plot([1:niter], lb_log[1:niter]; color=[1., 0., 0.]); title("Lower bound")
subplot(212); plot(1:niter, tl_log[1:niter]; color=[0., 0., 1.]); title("Predictive likelihood")

figure()
colors = ["c","b","g","r","m","k","y","w"]
for k = 1:T
    inds = ass .== k
    plot3D(vec(X[1,inds]), vec(X[2,inds]),vec(X[3,inds]),".", color=colors[k%length(colors)+1], markersize=10)
end
title("Point with color coded cluster assignment")
show()
