module SVDInplace

using LinearAlgebra, Random
import LinearAlgebra:
  require_one_based_indexing, chkstride1, BlasInt, LAPACK.chkargsok, LAPACK.getrf!
import LinearAlgebra.BLAS.libblastrampoline
import LinearAlgebra.BLAS.@blasfunc

gesv = :zgesvd_
@eval begin
  function zgesvd!(JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, RWORK, INFO)
    ccall(
      (@blasfunc($gesv), libblastrampoline),
      Cvoid,
      (
        Ref{Int8},   # JOBU
        Ref{Int8},   # JOBVT
        Ref{Int64}, # M
        Ref{Int64}, # N
        Ptr{ComplexF64}, # A
        Ref{Int64},   # LDA
        Ptr{Float64}, # S
        Ptr{ComplexF64}, # U
        Ref{Int64},   # LDU
        Ptr{ComplexF64}, # VT
        Ref{Int64},   # LDVT
        Ptr{ComplexF64}, # WORK
        Ref{Int64},   # LWORK
        Ptr{Float64}, # RWORK
        Ref{Int64}, # INFO
      ),
      JOBU,
      JOBVT,
      M,
      N,
      A,
      LDA,
      S,
      U,
      LDU,
      VT,
      LDVT,
      WORK,
      LWORK,
      RWORK,
      INFO,
    )
    return nothing
  end
end

function zgesvd_hworkspace(A)
  JOBU = Cchar('A')
  JOBVT = Cchar('A')
  M = size(A, 1)
  N = size(A, 2)
  LDA = size(A, 1)
  S = zeros(min(M, N))
  U = zeros(ComplexF64, M, M)
  LDU = size(U, 1)
  VT = zeros(ComplexF64, N, N)
  LDVT = size(VT, 1)
  LWORK = 10 * max(1, 2 * min(M, N) + max(M, N))
  WORK = zeros(ComplexF64, LWORK)
  RWORK = zeros(Float64, 5 * min(M, N))
  INFO = 0
  return JOBU, JOBVT, M, N, LDA, S, U, LDU, VT, LDVT, LWORK, WORK, RWORK, INFO
end

gesd = :zgesdd_
@eval begin
  function zgesdd!(JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, RWORK, IWORK, INFO)
    ccall(
      ($gesd, libblastrampoline),
      Cvoid,
      (
        Ref{Int8},   # JOBZ
        Ref{Int64}, # M
        Ref{Int64}, # N
        Ptr{ComplexF64}, # A
        Ref{Int64},   # LDA
        Ptr{Float64}, # S
        Ptr{ComplexF64}, # U
        Ref{Int64},   # LDU
        Ptr{ComplexF64}, # VT
        Ref{Int64},   # LDVT
        Ptr{ComplexF64}, # WORK
        Ref{Int64},   # LWORK
        Ptr{Float64}, # RWORK
        Ptr{Int64}, # RWORK
        Ref{Int64}, # INFO
      ),
      JOBZ,
      M,
      N,
      A,
      LDA,
      S,
      U,
      LDU,
      VT,
      LDVT,
      WORK,
      LWORK,
      RWORK,
      IWORK,
      INFO,
    )
    return nothing
  end
end

function zgesdd_hworkspace(A)
  JOBZ = Cchar('A')
  M = size(A, 1)
  N = size(A, 2)
  LDA = size(A, 1)
  S = zeros(min(M, N))
  U = zeros(ComplexF64, M, M)
  LDU = size(U, 1)
  VT = zeros(ComplexF64, N, N)
  LDVT = size(VT, 1)
  mx = max(M, N)
  mn = min(M, N)
  LWORK = max(2 * mn * mn + 2 * mn + mx)
  WORK = zeros(ComplexF64, LWORK)
  LRWORK = max(5 * mn * mn + 4 * mn, 2 * mx * mn + 2 * mn * mn + mn)
  RWORK = zeros(Float64, LRWORK)
  IWORK = zeros(Int64, 8 * min(M, N))
  INFO = [0]
  return JOBZ, M, N, LDA, S, U, LDU, VT, LDVT, LWORK, WORK, LRWORK, RWORK, IWORK, INFO
end

function zgesvd_simple(A, WS)
  JOBU, JOBVT, M, N, LDA, S, U, LDU, VT, LDVT, LWORK, WORK, RWORK, INFO = WS
  zgesvd!(JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, RWORK, INFO)
end

end # module
