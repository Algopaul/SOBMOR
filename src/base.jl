using LinearAlgebra
import Base

abstract type IterType end

struct FunMatrix <: IterType
  fun::Function
  upd::Function
  tan::Function
  source_vector::AbstractVector
  target_matrix::AbstractMatrix
  nparams::Int
end

function basemat(upd, tan, source_vector, target_matrix, nparams)
  function fun(theta)
    source_vector .= theta
    upd()
    return target_matrix
  end
  return FunMatrix(fun, upd, tan, source_vector, target_matrix, nparams)
end

function get_psd(
  n,
  dtype=Float64,
  source_vector=zeros(div(n * (n + 1), 2)),
  target_matrix=zeros(dtype, n, n),
)
  workspace_matrix = LowerTriangular(zeros(dtype, n, n))
  deriv_ws = zeros(complex(dtype), n)

  function upd()
    vector_to_lower_triang!(workspace_matrix, source_vector)
    mul!(target_matrix, workspace_matrix', workspace_matrix)
    return nothing
  end

  function tan!(g, u, v)
    mul!(deriv_ws, workspace_matrix, u)
    lower_triang_to_vector!(g, deriv_ws, v, adding=true, c1=true, c2=false)
    mul!(deriv_ws, workspace_matrix, v)
    lower_triang_to_vector!(g, deriv_ws, u, adding=true, c2=true)
    return nothing
  end
  return basemat(upd, tan!, source_vector, target_matrix, div(n * (n + 1), 2))
end

function get_skw(
  n,
  dtype=Float64,
  source_vector=zeros(div(n * (n - 1), 2)),
  target_matrix=zeros(dtype, n, n),
)
  workspace_matrix = LowerTriangular(zeros(dtype, n, n))

  function upd()
    workspace_matrix .= 0
    vector_to_strictly_lower_triang!(workspace_matrix, source_vector)
    target_matrix .= workspace_matrix .- workspace_matrix'
    return nothing
  end

  function tan!(g, u, v)
    strictly_lower_triang_to_vector!(g, u, v, adding=true, c1=true)
    strictly_lower_triang_to_vector!(g, v, u, subtracting=true, c2=true)
  end

  return basemat(upd, tan!, source_vector, target_matrix, div(n * (n - 1), 2))
end

function get_rct(
  n,
  m,
  dtype=Float64,
  source_vector=zeros(n * m),
  target_matrix=zeros(dtype, n, m),
)
  function upd()
    target_matrix .= Base.ReshapedArray(source_vector, (n, m), ())
    return nothing
  end

  function tan!(g, u, v)
    i = 1
    @views for j = 1:m
      g[i:i+n-1] .= real.(u .* conj(v[j])) .+ g[i:i+n-1]
      i += n
    end
  end

  return basemat(upd, tan!, source_vector, target_matrix, n * m)
end

function get_diag(
  n,
  dtype=Float64,
  source_vector=zeros(dtype, 1),
  target_matrix=Diagonal(ones(dtype, n)),
)
  workspace_matrix = ones(n)

  function upd()
    target_matrix.diag .= source_vector .* workspace_matrix
    return nothing
  end

  function tan!(g, u, v)
    g[1] = real(dot(u, v)) + g[1]
  end
  return basemat(upd, tan!, source_vector, target_matrix, 1)
end

export get_psd, get_skw, get_rct, get_diag, FunMatrix
