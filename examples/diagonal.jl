using LinearAlgebra
using BlockFactorizations
using BenchmarkTools

d = 512
n, m = 16, 8
# testing Diagonal BlockFactorization
A = [Diagonal(randn(d)) for i in 1:n, j in 1:m]
B = BlockFactorization(A)

x = [randn(d) for _ in 1:m]
y = [zeros(d) for _ in 1:n]
@btime A*x
@btime mul!(y, A, x);

x = randn(d*m)
y = zeros(d*n)
@btime B*x;
@btime mul!(y, B, x);
