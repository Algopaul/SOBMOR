using LinearAlgebra

import LinearAlgebra:
  require_one_based_indexing,
  chkstride1,
  BlasInt,
  LAPACK.chkargsok,
  LAPACK.getrf!,
  LinearAlgebra.BLAS.libblastrampoline,
  LinearAlgebra.BLAS.@blasfunc,
  svd!

for (getrf, dtype) in [(:dgetrf_, Float64), (:zgetrf_, ComplexF64)]
  @eval begin
    function getrf!(A::AbstractMatrix{$dtype}, ipiv::Vector{BlasInt})
      require_one_based_indexing(A)
      chkstride1(A)
      m, n = size(A)
      @assert length(ipiv) >= min(m, n)
      lda = max(1, stride(A, 2))
      info = Ref{BlasInt}()
      ccall(
        (LinearAlgebra.BLAS.@blasfunc($getrf), libblastrampoline),
        Cvoid,
        (
          Ref{BlasInt},
          Ref{BlasInt},
          Ptr{ComplexF64},
          Ref{BlasInt},
          Ptr{BlasInt},
          Ptr{BlasInt},
        ),
        m,
        n,
        A,
        lda,
        ipiv,
        info,
      )
      chkargsok(info[])
      return A, ipiv, info[]
    end
  end
end

"""
getrf!(A, ipiv) -> (A, ipiv, info)

Compute the LU factorization of a general M-by-N matrix `A`.
The pivot-vector `ipiv` can be provided to avoid an allocation.
Included by package `KWLinalg`.
"""
function getrf! end

function get_lusolve(A::AbstractVecOrMat{TA}, B::AbstractVecOrMat{TB}) where {TA,TB}
  lu_M = deepcopy(A) |> Array
  lu_D = lu!(lu_M)
  # TODO: Use type promotion to avoid initial multiplication
  AinvB = deepcopy(A * B) |> Array

  function update(Anew)
    lu_M .= Anew
    getrf!(lu_D.factors, lu_D.ipiv)
  end

  function solve(A, B)
    update(A)
    ldiv!(AinvB, lu_D, B)
    return AinvB
  end

  function solve(B)
    ldiv!(AinvB, lu_D, B)
    return AinvB
  end

  return solve, lu_D
end

export get_lusolve
