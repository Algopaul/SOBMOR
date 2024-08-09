using Random

function get_stable_parametric(n_state, n_input, n_output, n_matrices, pmin, pmax)
  weight_vec = zeros(n_matrices)
  hf = get_hatfun(weight_vec, pmin, pmax)

  # Building Blocks
  J = []
  R = []
  Q = []
  B = []
  C = []
  FT = []
  for _ = 1:n_matrices
    push!(J, get_skw(n_state))
    push!(R, get_psd(n_state))
    push!(Q, get_psd(n_state))
    push!(B, get_rct(n_state, n_input))
    push!(C, get_rct(n_output, n_state, ComplexF64))
    push!(FT, get_rct(n_output, n_input))
  end
  sI = get_diag(n_state, ComplexF64)
  sI.fun(5 * im)
  J = weighted_sum(weight_vec, J, get_split(even_partition(n_matrices, J[1].nparams)...))
  R = weighted_sum(weight_vec, R, get_split(even_partition(n_matrices, R[1].nparams)...))
  Q = weighted_sum(weight_vec, Q, get_split(even_partition(n_matrices, Q[1].nparams)...))
  B = weighted_sum(weight_vec, B, get_split(even_partition(n_matrices, B[1].nparams)...))
  C = weighted_sum(weight_vec, C, get_split(even_partition(n_matrices, C[1].nparams)...))
  FT = weighted_sum(weight_vec, FT, get_split(even_partition(n_matrices, FT[1].nparams)...))
  seed = 0
  thetaJ, seed = init_rep(init_skew, [n_state], seed, 1e-3, n_matrices)
  thetaR, seed = init_rep(init_psd, [n_state], seed, 1e-3, n_matrices)
  thetaQ, seed = init_rep(init_psd, [n_state], seed, 1e-3, n_matrices)
  thetaB, seed = init_rep(init_rct, [n_state, n_input], seed, 1e-3, n_matrices)
  thetaC, seed = init_rep(init_rct, [n_output, n_state], seed, 1e-3, n_matrices)
  thetaFT, seed = init_rep(init_rct, [n_output, n_input], seed, 1e-3, n_matrices)
  theta0 = Array{Float64}([thetaC; thetaJ; thetaR; thetaQ; thetaB; thetaFT])

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
  hf_split = get_split(partitioner([JRQB_nparams + C.nparams, FT.nparams])...)
  HF = get_sum(H, FT, hf_split)

  fun_sI = sI.fun
  Dfun = D.compose
  DinvBfun = DinvB.compose
  Hfun = H.compose
  H_fullupdate = HF.fun
  HF_fun = HF.compose

  function update_s(s)
    fun_sI(s)
    Dfun()
    DinvBfun()
    Hfun()
    HF_fun()
    return HF.target_matrix
  end

  function update_p(p)
    hf(real(p))
    J.compose()
    R.compose()
    Q.compose()
    B.compose()
    C.compose()
    FT.compose()
    JmR.compose()
    JmRQ.compose()
  end

  function update_model(theta)
    H_fullupdate(theta)
  end
  return HF,
  update_s,
  update_p,
  update_model,
  theta0,
  (J, R, Q, B, C, FT, JmR, JmRQ, D, weight_vec)
end

function init_rep(fn, dims, seed, noise_level, n_repetitions)
  theta = []
  for _ = 1:n_repetitions
    theta = [theta; fn(dims, seed, noise_level)]
    seed = seed + 1
  end
  return theta, seed
end

function init_skew(dims, seed=0, noise_level=1e-3)
  n = dims[1]
  j_init = ones(div(n * (n - 1), 2))
  return j_init + noise_level * randn(MersenneTwister(seed), size(j_init)...)
end

function init_psd(dims, seed=0, noise_level=1e-3)
  n = dims[1]
  id_init = zeros(div(n * (n + 1), 2))
  TriangularReshapes.lower_triang_to_vector!(id_init, Array(I(n) * 1.0))
  return id_init + noise_level * randn(MersenneTwister(seed), size(id_init)...)
end

function init_rct(dims, seed=0, noise_level=1e-3)
  n, m = dims
  b_init = ones(n * m)
  return b_init + noise_level * randn(MersenneTwister(seed), size(b_init)...)
end

function get_hatfun(target_vector, pmin, pmax)
  n = length(target_vector)
  interval_length = (pmax - pmin) / (n - 1)
  positions = Vector(range(pmin, stop=pmax, length=n))
  a = pmin
  b = pmax
  function hatfun(x)
    target_vector .= 0
    if x == a
      target_vector[1] = 1
    elseif x == b
      target_vector[n] = 1
    else
      i1 = Int(ceil((x - a) / (b - a) * (n - 1)))
      target_vector[i1] = 1 - abs(positions[i1] - x) / interval_length
      if length(target_vector) > i1
        target_vector[i1+1] = 1 - abs(positions[i1+1] - x) / interval_length
      end
    end
    return target_vector
  end
  return hatfun
end

export get_stable_parametric, get_hatfun
