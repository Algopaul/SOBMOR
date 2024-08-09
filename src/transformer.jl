function get_tf_update(f1, transform)
  function update(theta)
    f1(theta)
    transform()
    return nothing
  end
  return update
end

function get_tf_tan(tan1, uvmod)
  function tan(g, u, v)
    um, vm = uvmod(u, v)
    tan1(g, um, vm)
    return nothing
  end
end

function transformer(A, target_matrix, transform, uvmod)
  upd = get_tf_update(A.fun, transform)
  fun = get_fun(upd, target_matrix)
  tan = get_tf_tan(A.tan, uvmod)
  return CompFunMatrix(fun, upd, tan, target_matrix, A.nparams, transform)
end

function get_transpose(A, target_matrix=Array(A.target_matrix'))
  AM = A.target_matrix

  function transform()
    target_matrix .= AM'
    return nothing
  end

  function uvmod(u, v)
    return v, u
  end
end
