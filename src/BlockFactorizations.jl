module BlockFactorizations

using LinearAlgebra
using Base.Threads
using Base.Threads: @spawn, @sync

const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}
const AbstractMatOrFacOrUni{T} = Union{AbstractMatrix{T}, Factorization{T}, UniformScaling{T}}
const AbstractVecOfVec{T} = AbstractVector{<:AbstractVector{T}}
const AbstractVecOfVecOrMat{T} = AbstractVector{<:AbstractVecOrMat{T}}

export BlockFactorization

include("block.jl")

end # module
