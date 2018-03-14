# LTVModels

[![Build Status](https://travis-ci.org/baggepinnen/LTVModels.jl.svg?branch=master)](https://travis-ci.org/baggepinnen/LTVModels.jl)

2018-03-14: More documentation and instructions to appear shortly. If you want to use this package before documentation is complete, feel free to open an issue and I'll help you out.

This repository implements the system-identification methods presented in  
[Bagge Carlson, F.](https://www.control.lth.se/Staff/FredrikBaggeCarlson.html), Robertsson, A. & Johansson, R.  
["Identification of LTV Dynamical Models with Smooth or Discontinuous Time Evolution by means of Convex Optimization"](https://arxiv.org/abs/1802.09794) (IEEE ICCA 2018).

# Installation
```julia
Pkg.clone("https://github.com/baggepinnen/LTVModelsBase.jl")
Pkg.clone("https://github.com/baggepinnen/LTVModels.jl")
using LTVModels
```

# Usage
Usage of many of the functions is demonstrated in `tests/runtests.jl`

Code to reproduce Fig. 1 in the paper

```julia
using LTVModels, Plots
gr(size=(400,300))
T_       = 400
x,xm,u,n,m = LTVModels.testdata(T_)

anim = Plots.Animation()
function callback(m)
    fig = plot(flatten(m.At), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients", show=true, ylims=(-0.05, 1))
    frame(anim, fig)
end

λ = 17
@time model = LTVModels.fit_statespace_admm(xm, u, λ, extend=true,
                                                iters    = 10000,
                                                D        = 1,
                                                zeroinit = true,
                                                tol      = 1e-5,
                                                ridge    = 0,
                                                cb       = callback)
gif(anim, "admm.gif", fps = 10)
y = predict(model,x,u)
e = x[:,2:end] - y[:,1:end-1]
println("RMS error: ",rms(e))

At,Bt = model.At,model.Bt
plot(flatten(At), l=(2,:auto), xlabel="Time index", ylabel="Model coefficients")
plot!([1,T_÷2-1], [0.95 0.1; 0 0.95][:]'.*ones(2), l=(:dash,:black, 1))
plot!([T_÷2,T_], [0.5 0.05; 0 0.5][:]'.*ones(2), l=(:dash,:black, 1), grid=false)
gui()
```
![window](figures/admm.gif)


## Kalman smoother
Code to fit a model by solving (7) using a Kalman smoother
```julia
using LTVModels, Plots
T = 2_000
A,B,x,u,n,m,N = LTVModels.testdata(T=T, σ_state_drift=0.001, σ_param_drift=0.001)

gr(size=(400,300))
anim = @animate for r2 = logspace(-3,3,10)
    R1          = 0.001*eye(n^2+n*m) # Increase for faster adaptation
    R2          = r2*eye(n)
    P0          = 10000R1
    model = fit_model(KalmanModel, copy(x),copy(u),R1,R2,P0,extend=true)

    plot(flatten(A), l=(2,), xlabel="Time index", ylabel="Model coefficients", lab="True", c=:red)
    plot!(flatten(model.At), l=(2,), lab="Estimated", c=:blue)
end
gif(anim, "kalman.gif", fps = 5)
```
![window](figures/kalman.gif)

## Dynamic programming solver
To solve the optimization problem in section IID, see the function `fit_statespace_dp` with usage example in the function [`benchmark_ss`](https://github.com/baggepinnen/LTVModels.jl/blob/master/src/seg_bellman.jl#L183)


## Two-step procedure
See functions in files `peakdetection.jl` and function `fit_statespace_constrained`

## Figs. 2-3
To appear

## Fig. 4
To appear

[DifferentialDynamicProgramming.jl](https://github.com/baggepinnen/DifferentialDynamicProgramming.jl/tree/dev)
