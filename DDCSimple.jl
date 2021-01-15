module DDCSimple

using Distributions

export lnwage, home, work, solveModel, CCP

struct Model
    N::Int64
    T::Int64
    β::Array{Float64,1}
    λ::Array{Float64,1}
    σ2::Float64
    ymin::Int64
    ymax::Int64
    xmin::Int64
    xmax::Int64
    Model(; N, T, β, λ, σ2, ymin, ymax, xmin, xmax) = new(N, T, β, λ, σ2, ymin, ymax, xmin, xmax)
end

const γ = 0.52277
const δ = 0.95

function unpack(model)
    N = model.N
    T = model.T
    β = model.β
    λ = model.λ
    σ2 = model.σ2
    return N, T, β, λ, σ2
end

function lnwage(x, β)
    return β[1] + β[2]*x + β[3]*x^2
end

function work(y, x, β)
    return y + lnwage(x, β)
end

function home(y, λ)
    return λ[1] + (1 + λ[2])*y
end


function genData(N, T, σ2, ymin, ymax, xmin, xmax)
    N_ϵ = rand(Normal(0, σ2), N, T)
    y = rand(ymin:ymax, N)
    x = rand(xmin:xmax, N)
    return N_ϵ, y, x
end


function solveModel(T, yrange, xrange, λ, β, σ2)
    Emax = Array{eltype(β),3}(undef, T, length(yrange), length(xrange))
    yint = collect(yrange)
    for (index, value) in enumerate(xrange)
        arg = log(home.(yint, Ref(λ))) .- lnwage.(value, Ref(β))
        BigPhi = cdf.(Normal(0,1), arg ./ sqrt(σ2))
        BigPhi_Tilde = cdf.(Normal(0,1), (arg .- σ2) ./ sqrt(σ2))
        ratio = ((1 .- BigPhi_Tilde) ./ (1 .- BigPhi))
        Emax[T,:,index] .= (λ[1] .+ (1 .+ γ[2]) .* yint) .* BigPhi .+ (yint .+ ratio .* exp.(lnwage.(value, β) .+ 0.5*σ2)) .* (1 .- BigPhi)
    end
    for t in reverse(1:T-1)
        for (index, value) in enumerate(xrange)
            DEV = (Emax[t+1, :, index] .- Emax[t+1, :, max(index, index + 1)]) ./ (1 + δ)
            arg = log(home.(yint, Ref(λ)) .+ DEV) .- lnwage.(value, Ref(β))
            BigPhi = cdf.(Normal(0,1), arg ./ sqrt(σ2))
            BigPhi_Tilde = cdf.(Normal(0,1), (arg .- σ2) ./ sqrt(σ2))
            ratio = ((1 .- BigPhi_Tilde) ./ (1 .- BigPhi))
            Emax[t,:,index] .= (λ[1] .+ (1 .+ γ[2]) .* yint .+ (1 / (1+ δ)) .* Emax[t+1, :, index]) .* BigPhi
            Emax[t,:,index] .+= (yint .+ (1 / (1+ δ)) .* Emax[t+1, :, max(index, index + 1)] .+ ratio .* exp.(lnwage.(yint, value, β) .+ 0.5*σ2)) .* (1 .- BigPhi)
        end
    end
    return Emax
end

function genpath(T, β, λ, N_ϵ, y, x_init, Emax, yrange, xrange)
    choiceHist = zeros(Int64, T)
    stateHist = zeros(Int64, T) .+ x_init
    valueHist = Array{Float64,2}(undef, 2, T)
    for t = 1:T-1
        xpos = findfirst(isequal(stateHist[t]), xrange)
        ypos = findfirst(isequal(y), yrange)
        stateHist[t+1] = stateHist[t]
        valueHist[1,t] = home(y, λ) + δ * Emax[t+1, ypos, xpos]
        valueHist[2,t] = y + exp(lnwage(stateHist[t], β) + N_ϵ[t]) + δ * Emax[t+1, ypos, xpos + 1]
        if valueHist[2,t] > valueHist[1,t]
            stateHist[t+1] += 1
            choiceHist[t] = 1
        end
    end
    valueHist[1,T] = home(y, λ) + N_ϵ[1,T]
    valueHist[2,T] = work(y, stateHist[T], β) + N_ϵ[2,T]
    if valueHist[2,T] > valueHist[1,T]
        choiceHist[T] = 1
    end
    return stateHist, choiceHist
end

function genpathAlt(T, β, λ, N_ϵ, y, x_init, Emax, yrange, xrange)
    choiceHist = zeros(Int64, T)
    stateHist = zeros(Int64, T) .+ x_init
    #valueHist = Array{Float64,2}(undef, 2, T)
    for t = 1:T-1
        stateHist[t+1] = stateHist[t]
        #println(typeof(stateHist[t]))
        #@assert typeof(stateHist[t]) == Int64
        DEV = Emax
        if N_ϵ[t] > arg
            stateHist[t+1] += 1
            choiceHist[t] = 1
        end
    end
    if  CCP(stateHist[T], T, y, λ, β, T, Emax, yrange, xrange) > N_ϵ[T]
        choiceHist[T] = 1
    end
    return stateHist, choiceHist
end

function simulate(model)
    N, T, β, λ, σ2 = unpack(model)
    yrange = model.ymin:model.ymax
    xrange = model.xmin:(model.xmax + T)
    N_ϵ, y, x_init = genData(N, T, σ2, model.ymin, model.ymax, model.xmin, model.xmax)
    Emax = solveModel(T, yrange, xrange, λ, β, σ2)
    #stateHistory = Array{Array{Int64,1},1}(undef, N)
    stateHistory = [zeros(Int64, T) for i = 1:N]
    for i in 1:N
        stateHistory[i][1] = x_init[i]
    end
    choiceHistory = Array{Array{Int64,1},1}(undef, N)
    stateHistoryAlt = deepcopy(stateHistory)
    choiceHistoryAlt = deepcopy(choiceHistory)
    #eps = rand(Uniform(), N, T)
    for i = 1:N
        #println("before genpath, i = $i")
        stateHistory[i], choiceHistory[i] = genpath(T, β, λ, N_ϵ[i, :], y[i], x_init[i], Emax, yrange, xrange)
        #println("before genpathAlt, i = $i")
        #stateHistoryAlt[i], choiceHistoryAlt[i] = genpathAlt(T, β, λ, eps[i, :], y[i], x_init[i], Emax, yrange, xrange)
    end
    return stateHistory, choiceHistory, y, stateHistoryAlt, choiceHistoryAlt
end

end


#=

function llcontrib(choice, state, y, λ, β)
    contrib = 0.0
    for t = 1:ModelParams.T
        if choice[t] == 1
            contrib += log(CCP(state[t], t, y[1], λ, β))
        else
            contrib += log(1 - CCP(state[t], t, y[1], λ, β))
        end
    end
    return contrib
end

function loglike(choice, state, y, λ, β)
    ll = 0.0
    for i = 1:ModelParams.N
        ll += llcontrib(choice[i], state[i], y[i], λ, β)
    end
    return -ll
end

function unpack(param)
    λ = param[1:2]
    β = param[3:end]
    return λ, β
end

function wrapll(param)
    λ, β = unpack(param)
    ll = loglike(choiceHistory, stateHistory, y, λ, β)
    return ll
end


function StateFx(state::Int64, nchoice::Int64)
    output = Vector{Int64}(undef, nchoice)
    for i in 1:nchoice
        output[i] = state .+ 1
    #for i in 1:(nchoice-1)
    #    output[i] += 1
    end
    output[end] -= 1
    return output
end

function genStateSpace(init::Int64, T::Int64, nchoice::Int64)
    Domain_set = Dict{Int64,Vector{Int64}}()
    Domain_set[2] = StateFx(init, nchoice)
    for t = 2:T-1
        D = Vector{Int64}(undef,0)
        for i in Domain_set[t]
            #D = vcat(feasibleSet(i, T),D)
            append!(D, StateFx(i, nchoice)) #MUCH FASTER
        end
        Domain_set[t+1] = unique(D)
    end
    return Domain_set
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
