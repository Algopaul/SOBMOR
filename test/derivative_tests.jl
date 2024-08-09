function num_grad(f, theta, h=1e-6)
  g = similar(theta)
  for i in eachindex(theta)
    theta[i] += h
    g[i] = f(theta)
    theta[i] -= 2 * h
    g[i] = (g[i] - f(theta)) / (2 * h)
    theta[i] += h
  end
  return g
end

function uv_prod(f, u, v)
  return fun(param) = real(u' * f(param) * v)
end

function test_deriv(f, tan, shape, theta, debug=false)
  u = rand(ComplexF64, shape[1])
  v = rand(ComplexF64, shape[2])
  f(theta)
  g = zeros(size(theta))
  tan(g, u, v)
  g_num = num_grad(uv_prod(f, u, v), theta)
  if debug
    print(g)
    print(g_num)
  end
  @test g ≈ g_num rtol = 1e-6
  if debug
    print(norm(g - g_num))
  end
  nalloc_tan = @allocated tan(g, u, v)
  @test nalloc_tan == 0
  f(theta)
  nalloc_fun = @allocated f(theta)
  @test nalloc_fun == 0
end

function test_deriv(p::FunMatrix, debug=false)
  test_deriv(p.fun, p.tan, size(p.target_matrix), rand(p.nparams), debug)
end

function test_deriv(p::CompFunMatrix, debug=false, n_params=p.nparams)
  test_deriv(p.fun, p.tan, size(p.target_matrix), rand(n_params), debug)
end

@testset "Derivatives" begin
  test_deriv(get_psd(3))
  test_deriv(get_skw(3))
  test_deriv(get_rct(3, 4))
  test_deriv(get_diag(3))
end

@testset "Parametric" begin
  p1 = get_psd(5)
  p2 = get_psd(5)
  p3 = get_psd(5)
  weight_vec = zeros(3)
  hf = get_hatfun(weight_vec, 1.2, 4.5)
  hf(3.6)
  P = [p1, p2, p3]
  p = weighted_sum(weight_vec, P, get_split(even_partition(3, P[1].nparams)...))
  test_deriv(p)
end
