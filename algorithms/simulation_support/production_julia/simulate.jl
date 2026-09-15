# Standalone compiled adaptation of the manuscript's fixed-network Julia
# simulation. See README.md beside this file for provenance and differences.
using Random

function load_graph(path::String, n::Int)
    n >= 2 || error("At least two nodes are required")
    adjacency = [Int[] for _ in 1:n]
    open(path) do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            u, v = parse.(Int, split(strip(line))) .+ 1
            1 <= u < v <= n || error("Expected canonical 0-based edges with u < v")
            push!(adjacency[u], v)
            push!(adjacency[v], u)
        end
    end
    foreach(sort!, adjacency)
    all(!isempty(v) && length(unique(v)) == length(v) for v in adjacency) ||
        error("Isolated nodes or duplicate edges are unsupported")
    reached = falses(n)
    reached[1] = true
    queue = [1]
    for u in queue, v in adjacency[u]
        if !reached[v]
            reached[v] = true
            push!(queue, v)
        end
    end
    all(reached) || error("The graph must be connected")
    return adjacency
end

@inline function payoff_value(index, state, cooperative_neighbours, degree,
                              benefit, cost, average)
    raw = benefit * cooperative_neighbours[index] - cost * degree[index] * state[index]
    return average ? raw / degree[index] : raw
end

@inline function stable_logistic(z::Float64)
    if z >= 0.0
        return 1.0 / (1.0 + exp(-z))
    end
    ez = exp(z)
    return ez / (1.0 + ez)
end

@inline function choose_source(::Val{R}, target, state, cooperative_neighbours,
                               adjacency, degree, benefit, cost, delta, average,
                               buffer, rng) where R
    if R == :PC
        source = adjacency[target][rand(rng, 1:degree[target])]
        own = payoff_value(target, state, cooperative_neighbours, degree, benefit, cost, average)
        other = payoff_value(source, state, cooperative_neighbours, degree, benefit, cost, average)
        return rand(rng) < stable_logistic(delta * (other - own)) ? source : target
    end
    include_self = R == :IM
    offset = include_self ? 1 : 0
    max_score = -Inf
    if include_self
        max_score = delta * payoff_value(target, state, cooperative_neighbours, degree, benefit, cost, average)
        buffer[1] = max_score
    end
    @inbounds for k in eachindex(adjacency[target])
        j = adjacency[target][k]
        score = delta * payoff_value(j, state, cooperative_neighbours, degree, benefit, cost, average)
        buffer[k + offset] = score
        max_score = max(max_score, score)
    end
    total = 0.0
    @inbounds for k in 1:(degree[target] + offset)
        buffer[k] = exp(buffer[k] - max_score)
        total += buffer[k]
    end
    threshold = rand(rng) * total
    cumulative = include_self ? buffer[1] : 0.0
    include_self && threshold < cumulative && return target
    @inbounds for k in eachindex(adjacency[target])
        cumulative += buffer[k + offset]
        threshold < cumulative && return adjacency[target][k]
    end
    return adjacency[target][end]
end

@inline function update_once!(rule, state, cooperative_neighbours, adjacency,
                              degree, benefit, cost, delta, mu, average, buffer, rng)
    target = rand(rng, 1:length(state))
    source = choose_source(rule, target, state, cooperative_neighbours, adjacency,
                           degree, benefit, cost, delta, average, buffer, rng)
    new = rand(rng) < mu ? rand(rng, 0:1) : state[source]
    change = new - state[target]
    if change != 0
        state[target] = new
        @inbounds for neighbour in adjacency[target]
            cooperative_neighbours[neighbour] += change
        end
    end
    return change
end

function simulate(adjacency, rule::Val{R}, average::Bool, benefit::Float64,
                  cost::Float64, delta::Float64, mu::Float64, steps::Int,
                  burn_in::Int, seed::Int, initial_cooperation::Float64,
                  record_every::Int, trace_io) where R
    rng = Xoshiro(seed)
    n = length(adjacency)
    degree = length.(adjacency)
    state = Int.(rand(rng, n) .< initial_cooperation)
    cooperative_neighbours = [sum(state[j] for j in adjacency[i]) for i in 1:n]
    buffer = zeros(Float64, maximum(degree) + 1)
    count = sum(state)
    # Wide accumulators also cover N*T beyond the Int64 range.
    total = Int128(0)
    block_total = Int128(0)
    block_steps = 0
    started = time_ns()
    for step in 1:(burn_in + steps)
        count += update_once!(rule, state, cooperative_neighbours, adjacency,
                               degree, benefit, cost, delta, mu, average, buffer, rng)
        if step > burn_in
            measured = step - burn_in
            total += count
            block_total += count
            block_steps += 1
            if measured % record_every == 0 || measured == steps
                if trace_io !== nothing
                    println(trace_io, join((measured, Float64(block_total) / (n * Float64(block_steps)),
                                            Float64(total) / (n * Float64(measured))), ','))
                    flush(trace_io)
                end
                block_total = Int128(0)
                block_steps = 0
            end
        end
    end
    return (total, Float64(total) / (n * Float64(steps)), count / n,
            steps, burn_in + steps, (time_ns() - started) / 1e9)
end

function main(args)
    length(args) == 15 || error("Arguments: edges N output trace|- DB|PC|IM average|accumulated b c delta mu steps burn_in seed initial_cooperation record_every")
    edges, n_s, output, trace, rule_s, payoff, b_s, c_s, d_s, mu_s,
        steps_s, burn_s, seed_s, initial_s, every_s = args
    n, steps, burn, seed, every = parse.(Int, (n_s, steps_s, burn_s, seed_s, every_s))
    b, c, delta, mu, initial = parse.(Float64, (b_s, c_s, d_s, mu_s, initial_s))
    rule_s in ("DB", "PC", "IM") || error("Unsupported update rule")
    payoff in ("average", "accumulated") || error("Unsupported payoff aggregation")
    all(isfinite(x) && x >= 0 for x in (b, c, delta)) || error("Invalid payoff/selection parameters")
    0 <= mu <= 1 && 0 <= initial <= 1 || error("Invalid probability")
    steps > 0 && burn >= 0 && seed >= 0 && every > 0 || error("Invalid run length or seed")
    burn <= typemax(Int) - steps || error("Run length overflows Int")
    adjacency = load_graph(edges, n)
    rule = Val(Symbol(rule_s))
    trace_io = trace == "-" ? nothing : open(trace, "w")
    result = try
        trace_io !== nothing && println(trace_io, "measured_steps,block_mean,cumulative_mean")
        simulate(adjacency, rule, payoff == "average", b, c, delta, mu, steps, burn, seed, initial, every, trace_io)
    finally
        trace_io !== nothing && close(trace_io)
    end
    open(output, "w") do io
        println(io, "cooperation_sum,mean_cooperation,final_cooperation,measured_steps,total_updates,seconds")
        println(io, join(result, ','))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
