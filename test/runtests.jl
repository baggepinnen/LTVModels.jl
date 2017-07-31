using LTVModels
using Base.Test
using Plots
gr()
# LTVModels.test_gmmmodel() # Not working on 0.6 due to GaussianMixtures.jl
@test all(LTVModels.test_fit_statespace() .< 0.3)
LTVModels.benchmark_const(100, 2, true) # Dynamic Programming Bellman
LTVModels.benchmark_ss(100, 2, true)    # Dynamic Programming Bellman
LTVModels.test_kalmanmodel()
