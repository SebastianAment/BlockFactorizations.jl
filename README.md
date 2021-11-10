# BlockFactorizations.jl
This package contains a data structure that wraps a matrix of matrices or factorizations and acts like the matrix resulting from concatenating the input matrices (see below).
Notably, this allows the use of canonical linear algebra routines that just need to access `mul!` or `*` without special consideration for the block structure.
The structure contained herein differentiates itself from [BlockArrays.jl](https://github.com/JuliaArrays/BlockArrays.jl) 
in that it allows the blocks to be `AbstractMatrix` or `Factorization` types, therefore allowing the exploitation of more general matrix structure for efficient matrix multiplications.
See the [Wikipedia](https://en.wikipedia.org/wiki/Block_matrix) article for more information on block matrices.

## Basic Usage

### Block Matrix with Dense Blocks
Say we need to work with a block matrix `A` like the following
```julia
using LinearAlgebra
d = 512
n, m = 3, 4
A = [randn(d, d) for i in 1:n, j in 1:m]
```
Then Julia already allows for convenient syntax for matrix-vector multiplication,
by using a vector of vectors like so
```julia
using BenchmarkTools
x = [randn(d) for _ in 1:m]
y = [zeros(d) for _ in 1:n]
@btime A*x
  12.652 ms (321 allocations: 1.29 MiB)
@btime mul!(y, A, x);
  12.700 ms (320 allocations: 1.29 MiB)
```
However, we see that even `mul!` allocates significant memory since `mul!` is not applied recursively.
Instead `*` is used to multiply the vector elements.

With this package, we can wrap `A` to act like a normal matrix for multiplication: 
```julia
using BlockFactorizations
B = BlockFactorization(A)
x = randn(d*m)
y = randn(d*n)
@btime B*x;
  11.589 ms (30 allocations: 67.12 KiB)
@btime mul!(y, B, x);
  11.691 ms (28 allocations: 3.05 KiB)
```
Notably, this allows us to use canonical linear algebra routines that just need to access `mul!` or `*` without special consideration for the block structure.

### Block Matrix with Diagonal Blocks
Instead of having dense blocks, we can have more general block types, which can allow for more efficient matrix multiplication:
```julia
A = [Diagonal(randn(d)) for i in 1:n, j in 1:m]
B = BlockFactorization(A)

x = [randn(d) for _ in 1:m]
y = [zeros(d) for _ in 1:n]
@btime A*x
  117.001 μs (321 allocations: 1.29 MiB)
@btime mul!(y, A, x);
  127.356 μs (320 allocations: 1.29 MiB)
  
x = randn(d*m)
y = zeros(d*n)
@btime B*x;
  58.703 μs (158 allocations: 81.12 KiB)
@btime mul!(y, B, x);
  45.743 μs (156 allocations: 17.05 KiB)
```
Note that as long as `mul!` and `size` is defined for the element types, `BlockFactorization` will be applicable.

### Block Diagonal Matrix with Dense Blocks
Right above, we saw that BlockFactorizations supports `Diagonal` elements.
Further, it also specializes matrix multiplication in case of a [block diagonal matrix](https://en.wikipedia.org/wiki/Block_matrix#Block_diagonal_matrices):
```julia
d = 512
n, m = 16, 8
# testing Diagonal BlockFactorization
A = [Diagonal(randn(d)) for i in 1:n, j in 1:m]
B = BlockFactorization(A)

x = [randn(d) for _ in 1:m]
y = [zeros(d) for _ in 1:n]
@btime A*x
  135.849 μs (321 allocations: 1.29 MiB)
@btime mul!(y, A, x);
  135.590 μs (320 allocations: 1.29 MiB)

x = randn(d*m)
y = zeros(d*n)
@btime B*x;
  58.371 μs (157 allocations: 81.09 KiB)
@btime mul!(y, B, x);
  50.194 μs (156 allocations: 17.05 KiB)
```

## Notes
Note that the matrix multiplication function is parallelized with `@threads`.
Also, a BlockFactorization `B` can be converted to a non-blocked matrix using `Matrix(B)`.
Reported timings were computed on a 2017 MacBook Pro with a dual core processor and 16GB of RAM.
