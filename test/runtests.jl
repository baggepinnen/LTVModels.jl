using LTVModels
using Base.Test
using Plots
gr()
# LTVModels.test_gmmmodel()
LTVModels.benchmark_const(100, 2) # Dynamic Programming Bellman
LTVModels.benchmark_ss(100, 2)    # Dynamic Programming Bellman
LTVModels.test_kalmanmodel()
@test LTVModels.test_fit_statespace(1) < 0.5
