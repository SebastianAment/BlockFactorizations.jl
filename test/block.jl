module TestBlock
using LinearAlgebra
using BlockFactorizations
using Test

@testset "block" begin
    di = 2
    dj = 3
    n = 5
    m = 7
    element_types = (Float32, Float64, ComplexF32, ComplexF64)
    for elty in element_types
        # strided block matrix
        A = [randn(elty, di, dj) for i in 1:n, j in 1:m]
        # test general constructor
        F = BlockFactorization(A)
        @test size(F) == (n*di, m*dj)
        @test F.nindices isa AbstractRange # testing that stridedness was correctly identified
        @test F.mindices isa AbstractRange
        M = Matrix(F)
        @test eltype(F) == elty
        @test eltype(M) == elty

        for ni in 2:2, mi in 1:1
            for i in 1:di, j in 1:dj
                @test F[di*(ni-1)+i, dj*(mi-1)+j] == A[ni, mi][i, j]
                @test M[di*(ni-1)+i, dj*(mi-1)+j] == A[ni, mi][i, j]
            end
        end
        # test strided constructor
        isstrided = true
        F = BlockFactorization(A, isstrided)
        M = Matrix(F)
        @test size(F) == (n*di, m*dj)
        for ni in 2:2, mi in 1:1
            for i in 1:di, j in 1:dj
                @test F[di*(ni-1)+i, dj*(mi-1)+j] == A[ni, mi][i, j]
                @test M[di*(ni-1)+i, dj*(mi-1)+j] == A[ni, mi][i, j]
            end
        end

        # matrix multiply
        x = randn(size(M, 2))
        @test M*x ≈ F*x
        k = 3
        X = randn(size(M, 2), k)
        @test M*X ≈ F*X

        # general
        n, m = 4, 5
        nindices = [1, 2, n+1]
        mindices = [1, 4, m+1]
        A = fill(randn(0, 0), 2, 2)
        A[1, 1] = randn(1, 3)
        A[1, 2] = randn(1, 2)
        A[2, 1] = randn(3, 3)
        A[2, 2] = randn(3, 2)
        F = BlockFactorization(A, nindices, mindices)
        M = Matrix(F)
        @test size(F) == (n, m)
        for i in 1:4, j in 1:5
            @test F[i, j] == M[i, j]
        end

        # matrix multiply
        x = randn(size(M, 2))
        @test M*x ≈ F*x
        k = 3
        X = randn(size(M, 2), k)
        @test M*X ≈ F*X

        # special case for diagonal block factorization
        n, m = 3, 5
        k = 7
        A = Diagonal([randn(elty, n, m) for _ in 1:k])
        F = BlockFactorization(A)
        M = Matrix(F)

        x = randn(k*m) # always test multiplying with real vector too
        y = F*x
        @test length(y) == n*k
        @test y ≈ M*x

        x = randn(elty, k*m)
        y = F*x
        @test length(y) == n*k
        @test y ≈ M*x

        # testing with vector elements
        n, m = 2, 5
        d = 3
        A = [randn(elty, d) for i in 1:n, j in 1:m]
        F = BlockFactorization(A)
        M = Matrix(F)
        x = randn(elty, m)
        y = F*x
        @test length(y) == d*n
        @test y ≈ M*x

        diagF = diag(F)
        @test length(diagF) == minimum(size(M))
        @test diagF ≈ diag(M)

        # TODO: non-strided test
        # A = [randn(elty, 2, 3) randn(elty, 2, 4);
        #     randn(elty, 3, 3) randn(elty, 3, 4)]
        # F = BlockFactorization(A)
        # M = Matrix(F)
        # x = randn(elty, m)
        # y = F*x
        # @test length(y) == d*n
        # @test y ≈ M*x
    end

end

end
