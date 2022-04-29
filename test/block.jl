module TestBlockFactorizations
using LinearAlgebra
using BlockFactorizations
using Test

@testset "block" begin

    element_types = (Float32, Float64, ComplexF32, ComplexF64)
    for elty_A in element_types
        @testset "eltype $elty_A" begin
            # strided block matrix
            di = 2
            dj = 3
            n = 5
            m = 7
            stride_matrices = [
                                [randn(elty_A, di, dj) for i in 1:n, j in 1:m], # generic strided matrix
                                Diagonal([randn(elty_A, di, dj) for i in 1:n]), # diagonal strided matrix
                                # IDEA: test with scalar elements as fallback
                              ]
            for A in stride_matrices
                # test general constructor
                di, dj = size(A[1])
                n, m = size(A)
                strided_factorizations = [BlockFactorization(A),
                                          BlockFactorization(A, isstrided = true),
                                          BlockFactorization(A, di, dj)]
                # stride_matrices = [A, A, A, D, D, D] # matrices corresponding to strided_factorizations
                for F in strided_factorizations
                    M = Matrix(F)
                    @test size(F) == (size(A, 1) * size(A[1], 1), size(A, 2) * size(A[1], 2)) # (n*di, m*dj)
                    @test F.nindices isa AbstractRange # testing that stridedness was correctly identified
                    @test F.mindices isa AbstractRange
                    @test eltype(F) == elty_A
                    @test eltype(M) == elty_A
                    for ni in 2:2, mi in 1:1
                        for i in 1:di, j in 1:dj
                            @test F[di*(ni-1)+i, dj*(mi-1)+j] == A[ni, mi][i, j]
                            @test M[di*(ni-1)+i, dj*(mi-1)+j] == A[ni, mi][i, j]
                        end
                    end
                    # matrix multiply
                    for elty_B in element_types
                        x = randn(elty_B, size(M, 2))
                        @test M*x ≈ F*x
                        k = 3
                        X = randn(elty_B, size(M, 2), k)
                        @test M*X ≈ F*X
                        # IDEA: test left multiply
                    end
                    # diagonal of factorization
                    diagF = diag(F)
                    @test length(diagF) == minimum(size(M))
                    @test diagF ≈ diag(M)
                end
                allo_1 = @allocated BlockFactorization(A, isstrided = true)
                allo_2 = @allocated BlockFactorization(A, isstrided = false)
                @test allo_1 < allo_2
            end

            # general, non-strided block matrix
            n, m = 4, 5 # this is the size of the non-block equivalent of the following matrices
            nindices = [1, 2, n+1]
            mindices = [1, 4, m+1]
            A = fill(randn(elty_A, 0, 0), 2, 2)
            A[1, 1] = randn(elty_A, 1, 3)
            A[1, 2] = randn(elty_A, 1, 2)
            A[2, 1] = randn(elty_A, 3, 3)
            A[2, 2] = randn(elty_A, 3, 2)
            # D = Diagonal([randn(elty_A, 1, 3), randn(elty_A, 3, 2)])
            matrices = [A, Diagonal(A)]

            for A in matrices
                factorizations = [BlockFactorization(A),
                                  BlockFactorization(A, isstrided = false),
                                  BlockFactorization(A, nindices, mindices)
                                  ]
                for F in factorizations
                    M = Matrix(F)
                    @test size(F) == size(M)
                    for i in 1:n, j in 1:m
                        @test F[i, j] == M[i, j]
                    end

                    # matrix multiply
                    for elty_B in element_types
                        x = randn(elty_B, size(M, 2))
                        @test M*x ≈ F*x
                        k = 3
                        X = randn(elty_B, size(M, 2), k)
                        @test M*X ≈ F*X
                        # IDEA: test left multiply
                    end
                end
            end

            # testing with vector elements
            n, m = 2, 5
            d = 3
            A = [randn(elty_A, d) for i in 1:n, j in 1:m]
            F = BlockFactorization(A)
            M = Matrix(F)
            for elty_B in element_types
                x = randn(elty_B, size(M, 2))
                @test M*x ≈ F*x
                k = 3
                X = randn(elty_B, size(M, 2), k)
                @test M*X ≈ F*X
                # IDEA: test left multiply
            end
            diagF = diag(F)
            @test length(diagF) == minimum(size(M))
            @test diagF ≈ diag(M)

            # testing with heterogeneous block types
            n = 3
            A11 = Diagonal(randn(elty_A, n))
            A12 = randn(elty_A, n, n)
            A21 = UpperTriangular(randn(elty_A, n, n))
            A22 = cholesky(Hermitian(A21'A21))
            A = reshape([A11, A12, A21, A22], 2, :)
            @test eltype(A) == Any
            B = BlockFactorization(A)
            @test eltype(B) == elty_A

            d, n = 2, 3
            A = Diagonal([randn(elty_A, d, d) for _ in 1:n])
            B = BlockFactorization(A)
            @test B isa BlockDiagonalFactorization
        end
    end
end

end # TestBlockFactorizations
