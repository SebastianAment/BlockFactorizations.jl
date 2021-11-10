using LinearAlgebra
using BlockFactorizations
using BenchmarkTools

d = 512
n = 16
# testing Diagonal BlockFactorization
A = Diagonal([(randn(d, d)) for i in 1:n])
B = BlockFactorization(A)

# x = [randn(d) for _ in 1:n]
# y = [zeros(d) for _ in 1:n]
# @btime A*x; # this doesn't work in 1.6
# @btime mul!(y, A, x);

x = randn(d*n)
y = zeros(d*n)
@btime B*x;
@btime mul!(y, B, x);
