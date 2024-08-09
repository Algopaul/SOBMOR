using LinearAlgebra
import Base

struct CompFunMatrix <: IterType
  fun::Function # fun as in FunMatrix
  upd::Function # update components
  tan::Function # tan function as in FunMatrix
  target_matrix::AbstractMatrix # workspace to write matrix
  nparams::Int # number of params as in FunMatrix
  compose::Function # how the components are composed to form the composition
end

function get_upd(idx_split, f1, f2, compose)
  function update(theta)
    t1, t2 = idx_split(theta)
    if length(t1) > 0
      f1(t1)
    end
    if length(t2) > 0
      f2(t2)
    end
    compose()
    return nothing
  end
  return update
end

function get_fun(update, target_matrix)
  function fun(theta)
    update(theta)
    return target_matrix
  end
  return fun
end

function get_tan(tan1, tan2, idx_split, uvmod)
  function tan(g, u, v)
    g1, g2 = idx_split(g)
    (u1, v1), (u2, v2) = uvmod(u, v)
    if length(g1) > 0
      tan1(g1, u1, v1)
    end
    if length(g2) > 0
      tan2(g2, u2, v2)
    end
    return nothing
  end
  return tan
end

function composer(A, B, target_matrix, compose, uvmod, idx_split, no_new_params=false)
  upd = get_upd(idx_split, A.fun, B.fun, compose)
  fun = get_fun(upd, target_matrix)
  tan = get_tan(A.tan, B.tan, idx_split, uvmod)
  if no_new_params
    nparams = 0
  else
    nparams = A.nparams + B.nparams
  end
  return CompFunMatrix(fun, upd, tan, target_matrix, nparams, compose)
end

@inbounds function get_prod(
  A,
  B,
  idx_split,
  target_matrix=A.target_matrix * B.target_matrix,
  no_new_params=false,
)
  MA = A.target_matrix
  MB = B.target_matrix

  function compose()
    mul!(target_matrix, MA, MB)
    return nothing
  end

  Bv = zeros(ComplexF64, size(B.target_matrix, 1))
  utA = zeros(ComplexF64, size(A.target_matrix, 2))

  function uvmod(u, v)
    mul!(Bv, MB, v)
    mul!(utA, MA', u)
    return (u, Bv), (utA, v)
  end
  return composer(A, B, target_matrix, compose, uvmod, idx_split, no_new_params)
end

function get_sum(A, B, idx_split, target_matrix=A.target_matrix + B.target_matrix)
  MA = A.target_matrix
  MB = B.target_matrix

  function compose()
    target_matrix .= MA .+ MB
    return nothing
  end

  function uvmod(u, v)
    return (u, v), (u, v)
  end

  return composer(A, B, target_matrix, compose, uvmod, idx_split)
end

function get_diff(A, B, idx_split, target_matrix=A.target_matrix - B.target_matrix)
  MA = A.target_matrix
  MB = B.target_matrix

  function compose()
    target_matrix .= MA .- MB
    return nothing
  end

  mU = zeros(ComplexF64, size(MA, 1))

  function uvmod(u, v)
    mU .= .-u
    return (u, v), (mU, v)
  end
  return composer(A, B, target_matrix, compose, uvmod, idx_split)
end

function get_ainvb(A, B, idx_split, target_matrix=A.target_matrix \ B.target_matrix)
  MA = A.target_matrix
  MB = B.target_matrix

  fdinvb, lu_D = get_lusolve(MA, MB)

  function compose()
    target_matrix .= fdinvb(MA, MB)
    return nothing
  end

  mBv = zeros(ComplexF64, size(MB, 1))
  mAinvBv = zeros(ComplexF64, size(MB, 1))
  utAinv = zeros(ComplexF64, size(MA, 2))

  function uvmod(u, v)
    mul!(mBv, MB, v, -1, 0)
    ldiv!(mAinvBv, lu_D, mBv)
    ldiv!(utAinv, lu_D', u)
    return (utAinv, mAinvBv), (utAinv, v)
  end

  return composer(A, B, target_matrix, compose, uvmod, idx_split)
end

function get_adjoint(A::FunMatrix, target_matrix=Array(A.target_matrix'))
  Afun = A.fun
  Ata = A.target_matrix
  Atan = A.tan

  function compose()
    target_matrix .= Ata'
  end

  function fun(theta)
    Afun(theta)
    compose()
    return target_matrix
  end

  n, m = size(A.target_matrix)

  vc = Vector{ComplexF64}(undef, n)
  uc = Vector{ComplexF64}(undef, m)

  function tan(g, u, v)
    vc .= conj.(v)
    uc .= conj.(u)
    Atan(g, vc, uc)
  end

  return CompFunMatrix(fun, compose, tan, target_matrix, 0, compose)
end

function partitioner(nparamsvec)
  A = []
  lp = 0
  for nparams in nparamsvec
    push!(A, (lp + 1, lp + nparams))
    lp = lp + nparams
  end
  return A
end

function even_partition(n_matrices, n_params)
  A = []
  lp = 0
  for _ = 1:n_matrices
    push!(A, (lp + 1, lp + n_params))
    lp = lp + n_params
  end
  return A
end

function get_split(idc1, idc2)
  function split(theta)
    t1 = view(theta, idc1[1]:idc1[2])
    t2 = view(theta, idc2[1]:idc2[2])
    return t1, t2
  end
  return split
end

function get_split(idc1, idc2, idc3)
  function split(theta)
    t1 = view(theta, idc1[1]:idc1[2])
    t2 = view(theta, idc2[1]:idc2[2])
    t3 = view(theta, idc3[1]:idc3[2])
    return t1, t2, t3
  end
  return split
end

# Todo: make this a macro for arbitrary index lengths
function get_split(idc1, idc2, idc3, idc4, idc5, idc6)
  function split(theta)
    t1 = view(theta, idc1[1]:idc1[2])
    t2 = view(theta, idc2[1]:idc2[2])
    t3 = view(theta, idc3[1]:idc3[2])
    t4 = view(theta, idc4[1]:idc4[2])
    t5 = view(theta, idc5[1]:idc5[2])
    t6 = view(theta, idc6[1]:idc6[2])
    return t1, t2, t3, t4, t5, t6
  end
  return split
end

export get_prod, get_sum, get_diff, get_ainvb, get_adjoint, CompFunMatrix
