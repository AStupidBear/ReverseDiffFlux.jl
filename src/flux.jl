Flux.data(x) = ReverseDiff.value(x)
Flux.param(x) = ReverseDiff.track(x)

track(m, tape = ReverseDiff.InstructionTape()) = fmap(x -> x isa AbstractArray ? ReverseDiff.track(x, tape) : x, m)
untrack(m) = fmap(Flux.data, m)

function Flux.Optimise.update!(opt, x, x̄)
    Δ = -Flux.Optimise.apply!(opt, Flux.data(x), Flux.data(x̄))
    Flux.data(x) .+= Flux.data(Δ)
    ReverseDiff.deriv(x) .= 0
    return x
end

function Flux.destructure(m)
    xs = []
    fmap(m) do x
        x isa AbstractArray && push!(xs, x)
        return x
    end
    θ = vcat(vec.(Flux.data.(xs))...)
    re = p -> Flux._restructure(m, p)
    return Flux.param(θ), re
end

