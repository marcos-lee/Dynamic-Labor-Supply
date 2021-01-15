module DDCestimate
include("DDCSimple.jl")
using Optim, .DDCSimple, Statistics, NLSolversBase

function llcontrib(choice, state, y, yrange, xrange, λ, β, T, Emax)
    contrib = 0.0
    for t = 1:T 
        if choice[t] == 1
            contrib += log(CCP(state[t], t, y, λ, β, T, Emax, yrange, xrange))
        else
            contrib += log(1 - CCP(state[t], t, y, λ, β, T, Emax, yrange, xrange))
        end
    end
    return contrib
end

function loglike(choice, state, y, yrange, xrange, λ, β, N, T)
    ll = 0.0
    Emax = solveModel(T, yrange, xrange, λ, β)
    for i = 1:N
        ll += llcontrib(choice[i], state[i], y[i], yrange, xrange, λ, β, T, Emax)
    end
    return -ll
end

function wrapll(param, choiceHistory, stateHistory, y, yrange, xrange, N, T)
    β = param[1:3]
    λ = param[4:5]
    ll = loglike(choiceHistory, stateHistory, y, yrange, xrange, λ, β, N, T)
    return ll
end

function estimate(stateHistory, choiceHistory, y, yrange, xrange; guess = [9.8, 1.13, -0.55, 2.50, 0.30])
    T = size(stateHistory[1])[1]
    N = size(stateHistory)[1]
    func = OnceDifferentiable(vars -> wrapll(vars, choiceHistory, stateHistory, y, yrange, xrange, N, T),
                           guess; autodiff=:forward)
    opt = optimize(func, guess, method = LBFGS(), show_trace=true)
    return opt, Optim.minimizer(opt)
end


end


#=



function unpack(param)
    λ = param[1:2]
    β = param[3:end]
    return λ, β
end


=#

#####################






#=
#Julia 1.0.2
function feasibleSet(state::Int64, nchoice::Int64)
    st = copy(state)
    if st < 12
        output = Vector{Int64}(undef,nchoice)
        for i in 1:nchoice
            output[i] = copy(st)
        end
        for i in 1:(nchoice-1)
            output[i] += 1
        end
    else
        output = [st]
    end
    return output
end

function StateSpace(st::Int64, T::Int64, nchoice::Int64)
    # Define an auxiliary vector to calculate the feasible set
    # It adds 1 to the state vector according to the action taken
    # First element: go to school
    # Second: Work at 1
    # Third: Work at 2
    # Stay at home
    Domain_set = Dict{Int64,Vector{Int64}}()
    Domain_set[2] = feasibleSet(st, nchoice)
    for t = 2:T-1
        D = Vector{Int64}(undef,0)
        for i in Domain_set[t]
            #D = vcat(feasibleSet(i, T),D)
            append!(D,feasibleSet(i, nchoice)) #MUCH FASTER
        end
        Domain_set[t+1] = unique(D)
    end
    return Domain_set
end


function EmaxT(StateSpaceT, MC_ϵ)
    Emax = Array{Float64}(undef,size(StateSpaceT,1))
    i = 1
    for ss in StateSpaceT
        r1 = exp(lnwage(ss, β)) .+ MC_ϵ
        r2 = home(y, γ)
        Emax[i] = mean(max.(r1,r2))
        i += 1
    end
    return Emax
end
=#
