################################################################################
# assumes A is a matrix of matrices or factorizations each of which has size d by d
# IDEA: add block matrix inverse, so far only interested in enabling efficient multiply for iterative methods
"""
```
BlockFactorization{T} <: Factorization{T}
```
Wrapper type for a matrix of matrices or factorizations `A` with `eltype(A) = T`,
which acts as if component matrices were concatenated to form a single matrix `B`
with `eltype(B) = T`.
"""
struct BlockFactorization{T, AT, U, V} <: Factorization{T}
    A::AT
    nindices::U
    mindices::V
end
function BlockFactorization(A::AbstractMatrix, nindices, mindices)
    T = block_eltype(A)
    BlockFactorization{T, typeof(A), typeof(nindices), typeof(mindices)}(A, nindices, mindices)
end
const StridedBlockFactorization = BlockFactorization{<:Any, <:Any, <:StepRange, <:StepRange}
const BlockDiagonalFactorization = BlockFactorization{<:Any, <:Diagonal, <:Any, <:Any}

# calculates the element type of the BlockFactorization with blocks given by A
block_eltype(A::AbstractMatrix{<:Number}) = eltype(A)
block_eltype(A::AbstractMatrix{<:AbstractMatrix}) = eltype(eltype(A))
block_eltype(A::AbstractMatrix{<:Factorization}) = eltype(eltype(A))
function block_eltype(A::AbstractMatrix)
    T = Union{} # bottom type
    for Ai in A
        T = promote_type(T, eltype(Ai))
    end
    return T
end

function BlockFactorization(A::AbstractMatrix, di::Int, dj::Int = di)
    BlockFactorization(A, 1:di:di*size(A, 1)+1, 1:dj:dj*size(A, 2)+1)
end

# strided = true assumes that A has strided block indices, i.e. every element has the same size
function BlockFactorization(A::AbstractMatrix; isstrided::Bool = false)
    if isstrided
        di, dj = size(A[1, 1])
        nind, mind = 1:di:di*size(A, 1)+1, 1:dj:dj*size(A, 2)+1
    else
        n, m = size(A)
        nind, mind = zeros(Int, n+1), zeros(Int, m+1)
        nind[1], mind[1] = 1, 1
        for i in 1:n
            ni = size(A[i, 1], 1)
            all(==(ni), (size(A[i, j], 1) for j in 1:m)) || throw(DimensionMismatch("elements in column $i do not all have the same size"))
            nind[i+1] = nind[i] + ni
        end
        for j in 1:m
            mj = size(A[1, j], 2)
            all(==(mj), (size(A[i, j], 2) for i in 1:n)) || throw(DimensionMismatch("elements in row $j do not all have the same size"))
            mind[j+1] = mind[j] + mj
        end
        nind, mind = strided(nind), strided(mind)
    end
    return BlockFactorization(A, nind, mind)
end

# if possible, convert x to a range, otherwise return x unchanged
function strided(x::AbstractVector)
    length(x) < 2 && return x
    d = x[2] - x[1]
    isstrided = true
    for i in 2:length(x)-1
        if !(x[i+1] - x[i] ≈ d)
            isstrided = false
            break
        end
    end
    isstrided ? range(1, step = d, length = length(x)) : x
end

# allowing AT to be Matrix of Vectors would not work well with blockmul!
# better to cast to Matrix of Matrices
function BlockFactorization(A::AbstractMatrix{<:AbstractVector})
    B = [reshape(aij, :, 1) for aij in A]
    BlockFactorization(B)
end

Base.size(B::BlockFactorization, i::Int) = (1 ≤ i ≤ 2) ? size(B)[i] : 1
Base.size(B::BlockFactorization) = B.nindices[end]-1, B.mindices[end]-1
function Base.Matrix(B::BlockFactorization)
    C = zeros(eltype(B), size(B))
    @threads for j in 1:size(B.A, 2)
        jd = B.mindices[j] : B.mindices[j+1]-1
        for i in 1:size(B.A, 1)
            id = B.nindices[i] : B.nindices[i+1]-1
            M = Matrix(B.A[i, j])
            C[id, jd] .= M
        end
    end
    return C
end

# fallback for now
# Base.getindex(B::BlockFactorization, i::Int, j::Int) = Matrix(B)[i, j]
function Base.getindex(B::BlockFactorization, i::Int, j::Int)
    ni = findlast(≤(i), B.nindices)
    ni = isnothing(ni) ? 1 : ni
    nj = findlast(≤(j), B.mindices)
    nj = isnothing(nj) ? 1 : nj
    ri = i - B.nindices[ni] + 1
    rj = j - B.mindices[nj] + 1
    return B.A[ni, nj][ri, rj]
end

# IDEA: more efficient if neccesary
# function Base.getindex(B::StridedBlockFactorization, i::Int, j::Int)
#     println("strided")
#     ni, nj = mod1(i, B.nindices.step), mod1(j, B.mindices.step)
#     ri, rj = rem(i, B.nindices.step)+1, rem(j, B.mindices.step)+1
#     return B.A[ni, nj][ri, rj]
# end

function LinearAlgebra.diag(B::BlockFactorization)
    n = minimum(size(B))
    d = zeros(eltype(B), n)
    @threads for i in 1:n
        d[i] = B[i, i]
    end
    return d
end

############################## matrix multiplication ###########################
function Base.:*(B::BlockFactorization, x::AbstractVector)
    T = promote_type(eltype(B), eltype(x))
    y = zeros(T, size(B, 1))
    mul!(y, B, x)
end
function Base.:*(B::BlockFactorization, X::AbstractMatrix)
    T = promote_type(eltype(B), eltype(X))
    Y = zeros(T, size(B, 1), size(X, 2))
    mul!(Y, B, X)
end

function LinearAlgebra.mul!(y::AbstractVector, B::BlockFactorization, x::AbstractVector, α::Real = 1, β::Real = 0)
    xx = [@view x[B.mindices[i] : B.mindices[i+1]-1] for i in 1:length(B.mindices)-1]
    yy = [@view y[B.nindices[i] : B.nindices[i+1]-1] for i in 1:length(B.nindices)-1]
    blockmul!(yy, B.A, xx, α, β)
    return y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, B::BlockFactorization, X::AbstractMatrix, α::Real = 1, β::Real = 0)
    XX = [@view X[B.mindices[i] : B.mindices[i+1]-1, :] for i in 1:length(B.mindices)-1]
    YY = [@view Y[B.nindices[i] : B.nindices[i+1]-1, :] for i in 1:length(B.nindices)-1]
    blockmul!(YY, B.A, XX, α, β)
    return Y
end

# carries out multiplication for general BlockFactorization
function blockmul!(y::AbstractVecOfVecOrMat, G::AbstractMatrix, x::AbstractVecOfVecOrMat, α::Real = 1, β::Real = 0)
    @threads for i in eachindex(y)
        @. y[i] = β * y[i]
        for j in eachindex(x)
            mul!(y[i], G[i, j], x[j], α, 1)
        end
    end
    return y
end

################## specialization for diagonal block factorizations ############
# const DiagonalBlockFactorization{T} = BlockFactorization{T, <:Diagonal}
# carries out multiplication for Diagonal BlockFactorization
function blockmul!(y::AbstractVecOfVecOrMat, G::Diagonal,
                   x::AbstractVecOfVecOrMat, α::Real = 1, β::Real = 0)
    @threads for i in eachindex(y)
        mul!(y[i], G[i, i], x[i], α, β)
    end
    return y
end
