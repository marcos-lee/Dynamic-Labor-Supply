using Random, Revise

includet("DDCSimple.jl")
using .DDCSimple

Random.seed!(420)

opts = DDCSimple.Model(N = 10_000, T = 5, β = [98, 11.3, -5.5], λ = [25.0, 3.0])
stateHistory, choiceHistory, y = DDCSimple.simulate(opts)


includet("DDCestimate.jl")
using .DDCestimate

#opt = DDCestimate.estimate(stateHistory, choiceHistory, y)
DDCestimate.empiricalCCP(stateHistory, choiceHistory, 10_000, 5)

using Profile, ProfileView, BenchmarkTools, NLSolversBase
guess = [98, 11.3, -5.5, 25.0, 3.0]
opts = DDCSimple.Model(N = 10_000, T = 5, β = [98, 11.3, -5.5], λ = [25.0, 3.0])

res, xmin = DDCestimate.estimate(stateHistory, choiceHistory, y; guess = guess .- 5)
func = TwiceDifferentiable(vars -> DDCestimate.wrapll(vars, choiceHistory, stateHistory, y, 10_000, 5),
                           guess; autodiff=:forward)
numerical_hessian = hessian!(func,xmin)
var_cov_matrix = inv(numerical_hessian)
