@testset "Composition" begin
  p1 = get_psd(5)
  p2 = get_psd(5)
  p3 = get_psd(5)
  idcs = partitioner([p1.nparams, p2.nparams])
  split_fn = get_split(idcs[1], idcs[2])
  test_deriv(get_prod(p1, p2, split_fn))
  test_deriv(get_sum(p1, p2, split_fn))
  test_deriv(get_diff(p1, p2, split_fn))
  test_deriv(get_ainvb(p1, p2, split_fn))
  # test_deriv(weighted_sum([0.3, 0.7], (p1, p2), split_fn))
  theta1 = rand(p1.nparams)
  theta2 = rand(p2.nparams)
  theta3 = rand(p3.nparams)
  f1() = p1.fun(theta1)
  f2() = p2.fun(theta2)
  f3() = p3.fun(theta3)
  c = chainer([f1, f2, f3])
  c()
  nalloc = @allocated c()
  @test nalloc == 0
  # wm = weighted_sum([0.3, 0.7], (p1, p2), split_fn)
end

@testset "Adjoint" begin
  p1 = get_psd(5)
  p1t = get_adjoint(p1)
  r1 = get_rct(3, 4)
  r1t = get_adjoint(r1)
  test_deriv(p1t, false, p1.nparams)
  test_deriv(r1t, false, r1.nparams)
end

@testset "StableSystem" begin
  D, update_s, update_model = get_stable_mimo(10, 2, 2)
  theta = rand(D.nparams)
  f = D.fun
  test_deriv(D)
  update_s(5 * im)
  nalloc = @allocated update_s(6 * im)
  @test nalloc == 0
  update_model(theta)
  nalloc = @allocated update_model(theta)
  @test nalloc == 0
end

@testset "PHSystem" begin
  D, update_s, update_model = get_pH_system(5, 2, 2)
  theta = rand(D.nparams)
  f = D.fun
  test_deriv(D)
  update_s(5 * im)
  nalloc = @allocated update_s(6 * im)
  @test nalloc == 0
  update_model(theta)
  nalloc = @allocated update_model(theta)
  @test nalloc == 0
end

@testset "ParametricSystem" begin
  D, update_s, update_p, update_model = get_stable_parametric(5, 2, 3, 6, 0.4, 2.4)
  theta = rand(D.nparams)
  f = D.fun
  update_p(0.5)
  test_deriv(D)
  update_s(5 * im)
  nalloc_update_s = @allocated update_s(6 * im)
  @test nalloc_update_s == 0
  update_model(theta)
  nalloc = @allocated update_model(theta)
  @test nalloc == 0
end
