using ReverseDiff: TrackedArray, @grad
const data = ReverseDiff.value
const revtrack = ReverseDiff.track

nobacksies(f, x) = revtrack(nobacksies, f, x)
nobacksies(f, xs::Tuple) = map(x -> nobacksies(f, x), xs)
@grad nobacksies(f::Symbol, x) = data(x), Δ -> error("Nested AD not defined for $f")
@grad nobacksies(f::String, x) = data(x), Δ -> error(f)

softmax(xs::TrackedArray) = revtrack(softmax, xs)

@grad softmax(xs) = softmax(data(xs)), Δ -> (nobacksies(:softmax, ∇softmax(data(Δ), data(xs))),)

logsoftmax(xs::TrackedArray) = revtrack(logsoftmax, xs)

@grad logsoftmax(xs) = logsoftmax(data(xs)), Δ -> (nobacksies(:logsoftmax, ∇logsoftmax(data(Δ), data(xs))),)

depthwiseconv(x::TrackedArray, w::TrackedArray, cdims::DepthwiseConvDims; kw...) = revtrack(depthwiseconv, x, w, cdims; kw...)
depthwiseconv(x::AbstractArray, w::TrackedArray, cdims::DepthwiseConvDims; kw...) = revtrack(depthwiseconv, x, w, cdims; kw...)
depthwiseconv(x::TrackedArray, w::AbstractArray, cdims::DepthwiseConvDims; kw...) = revtrack(depthwiseconv, x, w, cdims; kw...)

@grad depthwiseconv(x, w, cdims::DepthwiseConvDims; kw...) =
  depthwiseconv(data(x), data(w), cdims; kw...),
    Δ -> nobacksies(:depthwiseconv,
      (NNlib.∇depthwiseconv_data(data.((Δ, w))..., cdims; kw...),
       NNlib.∇depthwiseconv_filter(data.((x, Δ))..., cdims; kw...),
       nothing))

conv(x::TrackedArray,  w::TrackedArray, cdims::DenseConvDims;  kw...) = revtrack(conv, x, w, cdims; kw...)
conv(x::AbstractArray, w::TrackedArray, cdims::DenseConvDims;  kw...) = revtrack(conv, x, w, cdims; kw...)
conv(x::TrackedArray,  w::AbstractArray, cdims::DenseConvDims; kw...) = revtrack(conv, x, w, cdims; kw...)

@grad conv(x, w, cdims::DenseConvDims; kw...) =
  conv(data(x), data(w), cdims; kw...),
    Δ -> nobacksies(:conv,
      (NNlib.∇conv_data(data.((Δ, w))..., cdims; kw...),
       NNlib.∇conv_filter(data.((x, Δ))..., cdims; kw...),
       nothing))

∇conv_data(x::TrackedArray,  w::TrackedArray, cdims::DenseConvDims;  kw...) = revtrack(∇conv_data, x, w, cdims; kw...)
∇conv_data(x::AbstractArray, w::TrackedArray, cdims::DenseConvDims;  kw...) = revtrack(∇conv_data, x, w, cdims; kw...)
∇conv_data(x::TrackedArray,  w::AbstractArray, cdims::DenseConvDims; kw...) = revtrack(∇conv_data, x, w, cdims; kw...)

@grad function ∇conv_data(y, w, cdims::DenseConvDims; kw...)
  return (
    ∇conv_data(data(y), data(w), cdims; kw...),
    Δ -> begin
      return nobacksies(:conv,
        (NNlib.conv(data.((Δ, w))..., cdims; kw...),
         NNlib.∇conv_filter(data.((Δ, y))..., cdims; kw...),
         nothing)
      )
    end
  )
end

maxpool(x::TrackedArray, pdims::PoolDims; kw...) = revtrack(maxpool, x, pdims; kw...)

@grad function maxpool(x, pdims::PoolDims; kw...)
  y = maxpool(data(x), pdims; kw...)
  y, Δ -> (nobacksies(:maxpool, NNlib.∇maxpool(data.((Δ, y, x))..., pdims; kw...)), nothing)
end

meanpool(x::TrackedArray, pdims::PoolDims; kw...) = revtrack(meanpool, x, pdims; kw...)

@grad function meanpool(x, pdims::PoolDims; kw...)
  y = meanpool(data(x), pdims; kw...)
  y, Δ -> (nobacksies(:meanpool, NNlib.∇meanpool(data.((Δ, y, x))..., pdims; kw...)), nothing)
end