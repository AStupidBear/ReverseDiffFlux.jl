module ReverseDiffFlux

using Flux, ReverseDiff
import Flux.NNlib: softmax, ∇softmax, logsoftmax, ∇logsoftmax, conv, ∇conv_data, depthwiseconv, maxpool, meanpool
import Flux.NNlib: DenseConvDims, DepthwiseConvDims, PoolDims
using MacroTools: @forward

include("nnlib.jl")
include("flux.jl")

function ReverseDiff.gradient(f, ps::Flux.Params)
    input = tuple(ps...)
    @assert !isempty(input)
    tape = input[1].tape
    @assert all(x -> x.tape === tape, input)
    cfg = ReverseDiff._GradientConfig(input, tape)
    output = ReverseDiff.gradient((a...) -> f(), input, cfg)
    return Flux.Zygote.Grads(Base.IdDict(zip(input, output)), ps)
end

function overload_gradient()
    @eval Flux.gradient(f, args...) = ReverseDiff.gradient(f, args...)
end

end
