@testset "SVD" begin
  A = rand(ComplexF64, 3, 3)
  B = deepcopy(A)
  JOBU, JOBVT, M, N, LDA, S, U, LDU, VT, LDVT, LWORK, WORK, RWORK, INFO =
    SVDInplace.zgesvd_hworkspace(A)
  SVDInplace.zgesvd!(
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
  _, SS, _ = svd(B)
  for (s, ss) in zip(S, SS)
    @test s == ss
  end
  A .= B
  k = @allocated SVDInplace.zgesvd!(
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
  @test k == 0
end
