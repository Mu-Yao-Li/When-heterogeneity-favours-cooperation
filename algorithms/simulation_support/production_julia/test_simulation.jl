using Test
include(joinpath(@__DIR__, "simulate.jl"))

@testset "DB/PC/IM transition probabilities" begin
    adjacency = [[2,3,4], [1,3], [1,2,5], [1,5], [3,4]]
    state0 = [1,0,1,0,0]
    degree = length.(adjacency)
    nc0 = [sum(state0[j] for j in adjacency[i]) for i in 1:5]
    b, c, delta, mu = 4.3, 1.0, 0.4, 0.05
    trials = 30000
    for average in (true, false), name in (:DB, :PC, :IM)
        raw = [b * nc0[i] - c * degree[i] * state0[i] for i in 1:5]
        f = average ? raw ./ degree : raw
        expected = zeros(5)
        for i in 1:5
            if name == :PC
                adoption = sum((state0[j] != state0[i]) /
                               (1 + exp(delta * (f[i] - f[j]))) for j in adjacency[i]) / degree[i]
            else
                candidates = name == :IM ? vcat([i], adjacency[i]) : adjacency[i]
                fitness = exp.(delta .* f[candidates])
                adoption = sum(fitness[k] for k in eachindex(candidates)
                               if state0[candidates[k]] != state0[i]) / sum(fitness)
            end
            expected[i] = (mu / 2 + (1 - mu) * adoption) / 5
        end
        rng = Xoshiro(73041)
        observed = zeros(Int, 5)
        buffer = zeros(maximum(degree) + 1)
        rule = Val(name)
        for _ in 1:trials
            state, nc = copy(state0), copy(nc0)
            change = update_once!(rule, state, nc, adjacency, degree, b, c,
                                  delta, mu, average, buffer, rng)
            @test sum(state) - sum(state0) == change
            for i in 1:5
                observed[i] += state[i] != state0[i]
            end
        end
        for i in 1:5
            se = sqrt(expected[i] * (1 - expected[i]) / trials)
            @test abs(observed[i] / trials - expected[i]) < 7se + 1 / trials
        end
    end
end

@testset "Integer neighbour counts, absorbing states and hold counting" begin
    adjacency = [[2,3,4], [1,3], [1,2,5], [1,5], [3,4]]
    degree = length.(adjacency)
    for average in (true, false), name in (:DB, :PC, :IM)
        rng = Xoshiro(3928)
        state = rand(rng, 0:1, 5)
        nc = [sum(state[j] for j in adjacency[i]) for i in 1:5]
        buffer = zeros(maximum(degree) + 1)
        rule = Val(name)
        for step in 1:1000
            update_once!(rule, state, nc, adjacency, degree, 4.3, 1.0,
                          0.4, 0.1, average, buffer, rng)
            @test nc == [sum(state[j] for j in adjacency[i]) for i in 1:5]
        end
        trace = IOBuffer()
        result = simulate(adjacency, rule, average, 4.0, 1.0, 0.01,
                           0.0, 23, 7, 42, 1.0, 10, trace)
        @test result[1] == 5 * 23
        @test result[2] == 1.0
        @test result[4] == 23 && result[5] == 30
        @test String(take!(trace)) == "10,1.0,1.0\n20,1.0,1.0\n23,1.0,1.0\n"
    end
    @test stable_logistic(60000.0) == 1.0
    @test stable_logistic(-60000.0) == 0.0
    @test stable_logistic(0.0) == 0.5
end
