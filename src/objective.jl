function l2_lift_objective(
  cfm::CompFunMatrix,
  update_funs::T,
  s_vals,
  targets,
  gamma,
  n_params=cfm.nparams - 1,
) where {T<:Tuple}
  if length(update_funs) == 3
    return l2_lift_objective_pm(cfm, update_funs..., s_vals, targets, gamma, n_params)
  elseif length(update_funs) == 2
    return l2_lift_objective(cfm, update_funs..., s_vals, targets, gamma, n_params)
  else
    throw(ErrorException("update_funs has wrong length"))
  end
end

function l2_lift_objective(
  cfm::CompFunMatrix,
  update_s,
  update_model,
  s_vals,
  targets,
  gamma,
  n_params=cfm.nparams - 1,
)
  @assert size(targets[1]) == size(cfm.target_matrix)
  err = im * cfm.target_matrix
  WS = SVDInplace.zgesvd_hworkspace(err)
  S = WS[6]
  U = WS[7]
  VT = WS[9]
  tan = cfm.tan
  GRAD = zeros(n_params)
  cfm_mat = cfm.target_matrix

  function lifted_l2(F, G, theta)
    if G !== nothing
      G .= 0
    end
    f = 0.0
    update_model(theta)
    for (s, target) in zip(s_vals, targets)
      update_s(s)
      err .= cfm_mat .- target
      SVDInplace.zgesvd_simple(err, WS)
      for (i, scrit) in enumerate(S)
        if scrit > gamma
          smg = scrit - gamma
          f += smg^2 / gamma
          if G !== nothing
            GRAD .= 0
            tan(GRAD, view(U, :, i), view(VT', :, i))
            G .+= 2 / gamma .* smg .* GRAD
          end
        end
      end
    end
    if F !== nothing
      return f
    end
  end

  return lifted_l2
end

function l2_lift_objective_pm(
  cfm::CompFunMatrix,
  update_s,
  update_p,
  update_model,
  s_vals,
  targets,
  gamma,
  n_params=cfm.nparams - 1,
)
  @assert size(targets[1]) == size(cfm.target_matrix)
  err = im * cfm.target_matrix
  WS = SVDInplace.zgesvd_hworkspace(err)
  S = WS[6]
  U = WS[7]
  VT = WS[9]
  tan = cfm.tan
  GRAD = zeros(n_params)
  cfm_mat = cfm.target_matrix

  function lifted_l2(F, G, theta)
    if G !== nothing
      G .= 0
    end
    f = 0.0
    update_model(theta)
    update_p(1.2)
    update_s(0.5im)
    p_prev = NaN
    for (s, target) in zip(s_vals, targets)
      if s[1] != p_prev
        update_p(s[1])
        p_prev = s[1]
      end
      update_s(s[2])
      err .= cfm_mat .- target
      SVDInplace.zgesvd_simple(err, WS)
      for (i, scrit) in enumerate(S)
        if scrit > gamma
          smg = scrit - gamma
          f += smg^2 / gamma
          if G !== nothing
            GRAD .= 0
            tan(GRAD, view(U, :, i), view(VT', :, i))
            G .+= 2 / gamma .* smg .* GRAD
          end
        end
      end
    end
    if F !== nothing
      return f
    end
  end

  return lifted_l2
end
