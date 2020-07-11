track(m, tape = ReverseDiff.InstructionTape()) = fmap(x -> x isa AbstractArray ? ReverseDiff.track(x, tape) : x, m)
untrack(m) = fmap(ReverseDiff.value, m)

function Flux.Optimise.update!(opt, x, x̄)
    Δ = -Flux.Optimise.apply!(opt, ReverseDiff.value(x), ReverseDiff.value(x̄))
    x_data = ReverseDiff.value(x)
    x_deriv = ReverseDiff.deriv(x)
    x_data .+= ReverseDiff.value(Δ)
    x_deriv .= 0
    return x
end

function Flux.destructure(m)
    xs = []
    fmap(m) do x
        x isa AbstractArray && push!(xs, x)
        return x
    end
    θ = vcat(vec.(ReverseDiff.value.(xs))...)
    re = p -> Flux._restructure(m, p)
    return Flux.param(θ), re
end
