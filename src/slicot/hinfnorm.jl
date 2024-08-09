using LinearAlgebra
using SLICOT_jll

function ab13dd!(
  DICO::Int8,
  JOBE::Int8,
  EQUIL::Int8,
  JOBD::Int8,
  N::Int,
  M::Int,
  P::Int,
  FPEAK::AbstractArray{Float64},
  A::AbstractArray{Float64},
  LDA::Int,
  E::AbstractArray{Float64},
  LDE::Int,
  B::AbstractArray{Float64},
  LDB::Int,
  C::AbstractArray{Float64},
  LDC::Int,
  D::AbstractArray{Float64},
  LDD::Int,
  GPEAK::AbstractArray{Float64},
  TOL::Float64,
  IWORK::AbstractArray{Int},
  DWORK::AbstractArray{Float64},
  LDWORK::Int,
  CWORK::AbstractArray{Complex{Float64}},
  LCWORK::Int,
  INFO::TINFO,
) where {TINFO}
  ccall(
    (:ab13dd_, libslicot),
    Cvoid,
    (
      Ref{Int8},
      Ref{Int8},
      Ref{Int8},
      Ref{Int8},
      Ref{Int},
      Ref{Int},
      Ref{Int},
      Ref{Float64},
      Ref{Float64},
      Ref{Int},
      Ref{Float64},
      Ref{Int},
      Ref{Float64},
      Ref{Int},
      Ref{Float64},
      Ref{Int},
      Ref{Float64},
      Ref{Int},
      Ref{Float64},
      Ref{Float64},
      Ref{Int},
      Ref{Float64},
      Ref{Int},
      Ref{Complex{Float64}},
      Ref{Int},
      Ref{Int},
    ),
    DICO,
    JOBE,
    EQUIL,
    JOBD,
    N,
    M,
    P,
    FPEAK,
    A,
    LDA,
    E,
    LDE,
    B,
    LDB,
    C,
    LDC,
    D,
    LDD,
    GPEAK,
    TOL,
    IWORK,
    DWORK,
    LDWORK,
    CWORK,
    LCWORK,
    INFO,
  )
end

"""
  lwork_ab13dd(n, p, m)

  Computes an upper bound for the workspaces required for ab13dd.

# Arguments
- N: Order of the system matrices E, A
- M: Number of inputs (size(B,2))
- P: Number of outputs (size(C,1))

"""
function lwork_ab13dd(n, p, m)
  ldwork = max(
    1,
    15 * n * n +
    p * p +
    m * m +
    (6 * n + 3) * (p + m) +
    4 * p * m +
    n * m +
    22 * n +
    7 * min(p, m),
  )
  lcwork = max(1, (n + m) * (n + p) + 2 * min(p, m) + max(p, m))
  return (ldwork, lcwork)
end

function linfnorm(A, B, C, D, E=Array{Float64}(I(size(A)[1])))
  E = E |> Array
  A = A |> Array
  B = B |> Array
  C = C |> Array
  D = D |> Array
  N = size(E, 1)
  M = size(B, 2)
  P = size(C, 1)
  FPEAK = [0.0, 1.0]
  GPEAK = [0.0, 1.0]
  TOL = 1e-12
  DICO = Cchar('C')
  JOBE = Cchar('G')
  EQUIL = Cchar('N')
  JOBD = Cchar('D')
  LDA = size(parent(A), 1)
  LDB = size(parent(B), 1)
  LDC = size(parent(C), 1)
  LDD = size(parent(D), 1)
  LDE = size(parent(E), 1)
  LDWORK, LCWORK = lwork_ab13dd(N, M, P)
  DWORK = Array{Float64}(undef, LDWORK)
  CWORK = Array{Complex{Float64}}(undef, LCWORK)
  LIWORK = N
  IWORK = Array{Int}(undef, LIWORK)
  INFO = [0]
  ab13dd!(
    DICO,
    JOBE,
    EQUIL,
    JOBD,
    N,
    M,
    P,
    FPEAK,
    A,
    LDA,
    E,
    LDE,
    B,
    LDB,
    C,
    LDC,
    D,
    LDD,
    GPEAK,
    TOL,
    IWORK,
    DWORK,
    LDWORK,
    CWORK,
    LCWORK,
    INFO,
  )
  if INFO != [0]
    return [NaN, NaN], [NaN, NaN]
  else
    norm_val = GPEAK[1]
    maximizer_freq = FPEAK[1]
    return norm_val, maximizer_freq
  end
end

function linferr(
  A1,
  B1,
  C1,
  D1,
  A2,
  B2,
  C2,
  D2;
  E1=Array{Float64}(I(size(A1)[1])),
  E2=Array{Float64}(I(size(A2)[1])),
)
  D = D1 - D2
  B = [B1; B2]
  C = [C1 -C2]
  n1 = size(A1)[1]
  n2 = size(A2)[1]
  A = [A1 zeros(n1, n2); zeros(n2, n1) A2]
  E = [E1 zeros(n1, n2); zeros(n2, n1) E2]
  return linfnorm(A, B, C, D, E)
end
