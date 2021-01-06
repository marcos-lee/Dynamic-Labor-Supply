module DDCestimate
include("DDCSimple.jl")
using Optim, .DDCSimple, Statistics, NLSolversBase

function CCP(state, t, y, λ, β, T)
    if t == T
        v1 = exp(work(y, state, β))
        v0 = exp(home(y, λ))
    else
        v1 = exp(work(y, state, β) + 0.95 * Emax(state + 1, y, λ, β))
        v0 = exp(home(y, λ) + 0.95 * Emax(state, y, λ, β))
    end
    ret = v1 / (v0 + v1)
    return ret
end

function llcontrib(choice, state, y, λ, β, T)
    contrib = 0.0
    for t = 1:T
        if choice[t] == 1
            contrib += log(CCP(state[t], t, y[1], λ, β, T))
        else
            contrib += log(1 - CCP(state[t], t, y[1], λ, β, T))
        end
    end
    return contrib
end

function loglike(choice, state, y, λ, β, N, T)
    ll = 0.0
    for i = 1:N
        ll += llcontrib(choice[i], state[i], y[i], λ, β, T)
    end
    return -ll
end

function wrapll(param, choiceHistory, stateHistory, y, N, T)
    β = param[1:3]
    λ = param[4:5]
    ll = loglike(choiceHistory, stateHistory, y, λ, β, N, T)
    return ll
end

function estimate(stateHistory, choiceHistory, y; guess = [98, 11.3, -5.5, 25.0, 3.0])
    T = size(stateHistory[1])[1]
    N = size(stateHistory)[1]
    func = OnceDifferentiable(vars -> wrapll(vars, choiceHistory, stateHistory, y, N, T),
                           guess; autodiff=:forward)
    opt = optimize(func, guess, method = NelderMead(), show_trace=true)
    return opt, Optim.minimizer(opt)
end


function empiricalCCP(stateHistory, choiceHistory, N, T)
    state = Array{Int64,2}(undef, N, T)
    choice = Array{Int64,2}(undef, N, T)
    for i = 1:N
        state[i,:] = stateHistory[i]
        choice[i,:] = choiceHistory[i]
    end
    ccp = Array{Array{Float64}}(undef, T)
    ccp[1] = [mean(choice[:,T])]
    for t = 2:T
        temp = Array{Float64}(undef, t)
        for x = 0:t-1
            temp[x+1] = mean(choice[x .== state[:,t], t])
        end
        ccp[t] = temp
    end
    return ccp
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
