using LinearAlgebra

function chain(f1, f2)
  function call()
    f1()
    f2()
    return nothing
  end
  return call
end

function chain_uv(f1, f2)
  function call(u, v)
    f1(u, v)
    f2(u, v)
    return nothing
  end
  return call
end

function chainer(fs)
  c = chain(fs[1], fs[2])
  for f in fs[3:end]
    c = chain(f, c)
  end
  return c
end

# TODO: replace this with macro for general weight lengths
function get_chain6(upds, compose, idx_split)
  f1 = upds[1]
  f2 = upds[2]
  f3 = upds[3]
  f4 = upds[4]
  f5 = upds[5]
  f6 = upds[6]
  function update(theta)
    ts = idx_split(theta)
    f1(ts[1])
    f2(ts[2])
    f3(ts[3])
    f4(ts[4])
    f5(ts[5])
    f6(ts[6])
    compose()
    return nothing
  end
end

function get_chain3(upds, compose, idx_split)
  f1 = upds[1]
  f2 = upds[2]
  f3 = upds[3]
  function update(theta)
    ts = idx_split(theta)
    f1(ts[1])
    f2(ts[2])
    f3(ts[3])
    compose()
    return nothing
  end
end

function get_tan3(tans, weights, idx_split)
  tan1 = tans[1]
  tan2 = tans[2]
  tan3 = tans[3]
  function tan(g, u, v)
    gs = idx_split(g)
    if weights[1] != 0
      tan1(gs[1], u, v)
      gs[1] .= weights[1] .* gs[1]
    end
    if weights[2] != 0
      tan2(gs[2], u, v)
      gs[2] .= weights[2] .* gs[2]
    end
    if weights[3] != 0
      tan3(gs[3], u, v)
      gs[3] .= weights[3] .* gs[3]
    end
    return nothing
  end
  return tan
end

function get_tan6(tans, weights, idx_split)
  tan1 = tans[1]
  tan2 = tans[2]
  tan3 = tans[3]
  tan4 = tans[4]
  tan5 = tans[5]
  tan6 = tans[6]
  function tan(g, u, v)
    gs = idx_split(g)
    if weights[1] != 0
      tan1(gs[1], u, v)
      gs[1] .= weights[1] .* gs[1]
    end
    if weights[2] != 0
      tan2(gs[2], u, v)
      gs[2] .= weights[2] .* gs[2]
    end
    if weights[3] != 0
      tan3(gs[3], u, v)
      gs[3] .= weights[3] .* gs[3]
    end
    if weights[4] != 0
      tan4(gs[4], u, v)
      gs[4] .= weights[4] .* gs[4]
    end
    if weights[5] != 0
      tan5(gs[5], u, v)
      gs[5] .= weights[5] .* gs[5]
    end
    if weights[6] != 0
      tan6(gs[6], u, v)
      gs[6] .= weights[6] .* gs[6]
    end
    return nothing
  end
  return tan
end

function weighted_sum(
  weights::Vector,
  fun_matrices,
  idx_split,
  target_matrix=Array(fun_matrices[1].target_matrix),
)
  TM = [fun_matrices[1].target_matrix]
  for fm in fun_matrices[2:end]
    fmt = fm.target_matrix
    push!(TM, fmt)
  end

  tans = []
  for fm in fun_matrices
    push!(tans, fm.tan)
  end
  tans = (tans...,)

  upds = []
  for fm in fun_matrices
    push!(upds, fm.fun)
  end
  upds = (upds...,)

  function compose()
    target_matrix .= 0
    for (i, weight) in enumerate(weights)
      if weight != 0.0
        BLAS.axpy!(weight, TM[i], target_matrix)
      end
    end
    return nothing
  end

  function get_tan()
    if length(weights) == 6
      return get_tan6(tans, weights, idx_split)
    elseif length(weights) == 3
      return get_tan3(tans, weights, idx_split)
    else
      error("tan for weight length not defined.")
    end
  end

  tan = get_tan()

  function get_update()
    if length(weights) == 6
      return get_chain6(upds, compose, idx_split)
    elseif length(weights) == 3
      return get_chain3(upds, compose, idx_split)
    else
      error("Chain for weight length not defined.")
    end
  end

  update = get_update()

  function fun(theta)
    update(theta)
    return target_matrix
  end

  nparams = 0
  for fm in fun_matrices
    nparams += fm.nparams
  end

  return CompFunMatrix(fun, update, tan, target_matrix, nparams, compose)
end

export weighted_sum, chainer
