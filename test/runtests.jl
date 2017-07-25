using LTVModels
using Base.Test
using Plots
gr()
# LTVModels.test_gmmmodel()
@test all(LTVModels.test_fit_statespace(10) .< 0.3)
LTVModels.benchmark_const(100, 2) # Dynamic Programming Bellman
LTVModels.benchmark_ss(100, 2)    # Dynamic Programming Bellman
LTVModels.test_kalmanmodel()
