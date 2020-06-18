# ReverseDiffFlux

[![Build Status](https://travis-ci.com/AStupidBear/ReverseDiffFlux.jl.svg?branch=master)](https://travis-ci.com/AStupidBear/ReverseDiffFlux.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/AStupidBear/ReverseDiffFlux.jl?svg=true)](https://ci.appveyor.com/project/AStupidBear/ReverseDiffFlux-jl)
[![Coverage](https://codecov.io/gh/AStupidBear/ReverseDiffFlux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/AStupidBear/ReverseDiffFlux.jl)

## Example

```julia
using Statistics
using ReverseDiffFlux
using Flux

x = randn(Float32, 10, 1, 100)
y = mean(x, dims = 1)

model = Chain(LSTM(10, 100), LSTM(100, 1)) |> ReverseDiffFlux.track

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
```
