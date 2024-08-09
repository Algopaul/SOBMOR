using SOBMOR
using Test
using LinearAlgebra

@testset "LU Solve" begin
  A = rand(5, 5)
  B = rand(5, 3)
  C = rand(5, 5)
  D = rand(5, 3)
  f, _ = get_lusolve(A, B)
  @test f(A, B) == A \ B
  nalloc = @allocated f(A, B)
  @test nalloc == 0
  @test f(C, D) == C \ D
end

include("./derivative_tests.jl")
include("./composition_tests.jl")
include("./svd_tests.jl")

@testset "l2gamma" begin
  D, update_s, update_model = get_pH_system(5, 2, 2)
  freqs = im * [0.1, 0.5]
  targets = [rand(ComplexF64, 2, 2), rand(ComplexF64, 2, 2)]
  loss = l2_lift_objective(D, update_s, update_model, freqs, targets, 0.5)
  grad = zeros(D.nparams - 1)
  theta = rand(D.nparams - 1)
  @allocated a = loss(nothing, grad, theta)
  k = (@allocated a = loss(nothing, grad, theta))
  @test k == 0
  pf(x) = loss(0, nothing, x)
  g = num_grad(pf, theta, 1e-6)
  loss(nothing, grad, theta)
  @test norm(g - grad) < 1e-5
end
