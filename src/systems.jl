function get_stable_mimo(n_state, n_input, n_output)
  # Building Blocks
  J = get_skw(n_state)
  R = get_psd(n_state)
  Q = get_psd(n_state)
  B = get_rct(n_state, n_input)
  C = get_rct(n_output, n_state, ComplexF64)
  sI = get_diag(n_state, ComplexF64)
  sI.fun(5 * im)

  # Compositions
  jmr_split = get_split(partitioner([J.nparams, R.nparams])...)
  JmR = get_diff(J, R, jmr_split)
  jmrq_split = get_split(partitioner([JmR.nparams, Q.nparams])...)
  JmRQ = get_prod(JmR, Q, jmrq_split)
  D = get_diff(sI, JmRQ, get_split((1, 0), (1, JmRQ.nparams)))
  dinvb_split = get_split(partitioner([JmRQ.nparams, B.nparams])...)
  DinvB = get_ainvb(D, B, dinvb_split)
  JRQB_nparams = J.nparams + R.nparams + Q.nparams + B.nparams
  h_split = get_split(partitioner([C.nparams, JRQB_nparams])...)
  H = get_prod(C, DinvB, h_split)

  fun_sI = sI.fun
  Dfun = D.compose
  DinvBfun = DinvB.compose
  Hfun = H.compose
  H_fullupdate = H.fun

  function update_s(s)
    fun_sI(s)
    Dfun()
    DinvBfun()
    Hfun()
    return H.target_matrix
  end

  function update_model(theta)
    H_fullupdate(theta)
  end

  return H, update_s, update_model
end

function get_pH_system(n_state, n_input, n_output=n_input)
  @assert n_output == n_input

  # Building Blocks
  J = get_skw(n_state)
  R = get_psd(n_state)
  Q = get_psd(n_state)
  B = get_rct(n_state, n_input)
  BT = get_adjoint(B, im * B.target_matrix')
  sI = get_diag(n_state, ComplexF64)
  sI.fun(5 * im)

  # Compositions
  jmr_split = get_split(partitioner([J.nparams, R.nparams])...)
  JmR = get_diff(J, R, jmr_split)
  jmrq_split = get_split(partitioner([JmR.nparams, Q.nparams])...)
  JmRQ = get_prod(JmR, Q, jmrq_split)
  D = get_diff(sI, JmRQ, get_split((1, 0), (1, JmRQ.nparams)))
  dinvb_split = get_split(partitioner([JmRQ.nparams, B.nparams])...)
  DinvB = get_ainvb(D, B, dinvb_split)
  JRQB_nparams = J.nparams + R.nparams + Q.nparams + B.nparams
  JRQn = J.nparams + R.nparams + Q.nparams
  Cs = [(JRQn + 1, JRQn + B.nparams), (JmR.nparams + 1, JmR.nparams + Q.nparams)]
  tmC = BT.target_matrix * Q.target_matrix
  C = get_prod(BT, Q, get_split(Cs...), tmC, true)
  A = [(1, JRQB_nparams), (1, JRQB_nparams)]
  h_split = get_split(A...)
  H = get_prod(C, DinvB, h_split)

  fun_sI = sI.fun
  Dfun = D.compose
  DinvBfun = DinvB.compose
  Hfun = H.compose
  H_fullupdate = H.fun

  function update_s(s)
    fun_sI(s)
    Dfun()
    DinvBfun()
    Hfun()
    return H.target_matrix
  end

  function update_model(theta)
    H_fullupdate(theta)
  end

  return H, update_s, update_model, (J, R, Q, B)
end

function get_pH_system_noQ(n_state, n_input, n_output=n_input)
  @assert n_output == n_input

  # Building Blocks
  J = get_skw(n_state)
  R = get_psd(n_state)
  B = get_rct(n_state, n_input)
  BT = get_adjoint(B, im * B.target_matrix')
  sI = get_diag(n_state, ComplexF64)
  sI.fun(5 * im)

  # Compositions
  jmr_split = get_split(partitioner([J.nparams, R.nparams])...)
  JmR = get_diff(J, R, jmr_split)
  D = get_diff(sI, JmR, get_split((1, 0), (1, JmR.nparams)))
  dinvb_split = get_split(partitioner([JmR.nparams, B.nparams])...)
  DinvB = get_ainvb(D, B, dinvb_split)
  JRB_nparams = J.nparams + R.nparams + B.nparams
  A = [(J.nparams + R.nparams + 1, JRB_nparams), (1, JRB_nparams)]
  h_split = get_split(A...)
  H = get_prod(BT, DinvB, h_split)

  fun_sI = sI.fun
  Dfun = D.compose
  DinvBfun = DinvB.compose
  Hfun = H.compose
  H_fullupdate = H.fun

  @inbounds function update_s(s)
    fun_sI(s)
    Dfun()
    DinvBfun()
    Hfun()
    return H.target_matrix
  end

  @inbounds function update_model(theta)
    H_fullupdate(theta)
  end

  return H, update_s, update_model, (J, R, B)
end

export get_stable_mimo, get_pH_system, get_pH_system_noQ
