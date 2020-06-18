using Test
using Statistics
using ReverseDiffFlux
using ReverseDiff
using Flux

@testset "ReverseDiffFlux.jl" begin
    x = rand(10, 10, 3, 3)
    ReverseDiff.gradient(x) do x
        sum(meanpool(x, (3, 3)))
    end

    m = ([1f0; 2f0], [2f0; 3f0]) |> ReverseDiffFlux.track
    @test ReverseDiff.gradient(Flux.params(m)) do
        sum(2 .* m[1] .+ m[2])
    end[m[1]] == [2f0; 2f0]

    x = randn(Float32, 10, 1, 100)
    y = mean(x, dims = 1)
    
    θ0, re = Flux.destructure(Dense(2, 1))
    @test ReverseDiff.gradient(θ0) do θ
        sum(re(θ)([0.5, 2.0]))
    end == [0.5, 2.0, 1]
    
    model = Chain(LSTM(10, 100), LSTM(100, 1)) |> ReverseDiffFlux.track
    θ0, re = Flux.destructure(model)
    @test Flux.destructure(re(θ0))[1] == θ0
    
    function loss(x, y)
        xs = Flux.unstack(x, 3)
        ys = Flux.unstack(y, 3)
        ŷs = model.(xs)
        l = 0f0
        for t in 1:length(ŷs)
            l += Flux.mse(ys[t], ŷs[t])
        end
        return l / length(ŷs)
    end
    ps = Flux.params(model)
    data = repeat([(x, y)], 100)
    opt = ADAMW(1e-3, (0.9, 0.999), 1e-4)
    cb = () -> Flux.reset!(model)
    ReverseDiffFlux.overload_gradient()
    Flux.@epochs 10 Flux.train!(loss, ps, data, opt, cb = cb)
    @test loss(x, y) < 0.02    
end
