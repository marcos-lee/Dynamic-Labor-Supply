using Random, Revise, NLSolversBase, LinearAlgebra

includet("DDCSimple.jl")
using .DDCSimple

Random.seed!(420)
println(1234)

opts = DDCSimple.Model(N = 10_000, T = 5, β = [12.1, 8.03, -1.05], λ = [2.50, 0.3],
    ymin=10, ymax=29, xmin=0, xmax=10)
stateHistory, choiceHistory, y, stateHistoryAlt, choiceHistoryAlt = DDCSimple.simulate(opts)


includet("DDCestimate.jl")
using .DDCestimate
println(1234)
guess = [12.1, 8.03, -1.05, 2.50, 0.30]
yrange = 10:29
xrange = 0:(10 + 5)

@code_warntype DDCestimate.loglike(choiceHistory, stateHistory, y, yrange, xrange, [2.50, 0.30], [12.1, 8.03, -1.05], 10_000, 5)

res, xmin = DDCestimate.estimate(stateHistoryAlt, choiceHistoryAlt, y, yrange, xrange; guess = guess)
func = TwiceDifferentiable(vars -> DDCestimate.wrapll(vars, choiceHistoryAlt, stateHistoryAlt, y, yrange, xrange, 10_000, 5),
                           guess; autodiff=:forward)
numerical_hessian = hessian!(func,xmin)
var_cov_matrix = inv(numerical_hessian)
sqrt.(diag(var_cov_matrix))

#=
a = DDCSimple.solveModel(opts)


Array{Int64,3}(undef, opts.T, opts.ymax-opts.ymin+1, opts.xmax-opts.xmin+opts.T+1)


N, T, β, λ, Emax = DDCSimple.unpack(opts)
yrange = opts.ymin:opts.ymax
xrange = opts.xmin:(opts.xmax + opts.T)
for (index, value) in enumerate(xrange)
    println(index)
    Emax[T,:,index] .= @. 0.5 + log(exp(DDCSimple.home(yrange, Ref(λ))) + exp(DDCSimple.work(yrange, Ref(value), Ref(β))))
end
N, T, β, λ, Emax = DDCSimple.unpack(opts)
yrange = opts.ymin:opts.ymax
xrange = opts.xmin:opts.xmax
for (index, value) in enumerate(xrange)
    Emax[T,:,index] .= @. γ + log(exp(home(yrange, Ref(λ))) + exp(work(yrange, value, Ref(β))))
end
for t in reverse(1:T-1)
    for (index, value) in enumerate(xrange)
        Emax[t,:,index] .= @. γ + log(exp(home(yrange, Ref(λ)) + δ * Emax[t+1,:, index]) + exp(work(yrange, value, Ref(β)) + δ * Emax[t+1,:, index]))
    end
end

DDCSimple.work.([1, 2, 3, 4], [4, 5, 6, 7], Ref([98, 11.3, -5.5]))

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
=#