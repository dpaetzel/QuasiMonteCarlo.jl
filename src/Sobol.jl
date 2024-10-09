"""
    SobolSample(R::RandomizationMethod = NoRand()) <: DeterministicSamplingAlgorithm

Samples taken from Sobol's base-2 sequence.
"""
Base.@kwdef @concrete struct SobolSample <: DeterministicSamplingAlgorithm
    R::RandomizationMethod = NoRand()
end

function sample(n::Integer, d::Integer, S::SobolSample, T = Float64)
    if n < 0
        throw(ArgumentError("number of samples must be non-negative"))
    end

    seq = Matrix{T}(undef, d, n)
    if n == 0
        return seq
    end

    # Use function barrier since `Sobol.SobolSeq(d)` can't be inferred
    return _sample!(seq, Sobol.SobolSeq(d), S.R)
end

function _sample!(seq::AbstractMatrix, s::Sobol.SobolSeq, R::RandomizationMethod)
    n = size(seq, 2)
    Sobol.skip!(s, n, @view(seq[:, begin]))
    for x in eachcol(seq)
        Sobol.next!(s, x)
    end
    return randomize(seq, R)
end

"""
Sample a Sobol' sequence into a preallocated array.
"""
function sample!(out::AbstractMatrix{T}, S::SobolSample) where {T <: AbstractFloat}
    d, n = size(out)
    if n == 0
        return out
    end
    seq = Sobol.SobolSeq(d)
    # https://github.com/JuliaMath/Sobol.jl/blob/685cec3fde77dce494c208f2de36c89f438254f6/src/Sobol.jl#L77
    skip(seq, n)
    for x in eachcol(out)
        Sobol.next!(seq, x)
    end

    randomize!(out, S.R)
    return out
end
