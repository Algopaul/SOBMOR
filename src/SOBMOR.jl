module SOBMOR

using LinearAlgebra
using DelimitedFiles
using Optim
using Printf
using ArgParse
import Base

include("./triangular_reshapes/TriangularReshapes.jl")
using .TriangularReshapes

include("./svd_inplace.jl")
using .SVDInplace

include("./slicot/hinfnorm.jl")

include("./mesh/adaptive_mesh.jl")
using .AdaptiveMesh

include("./timoshenko_beam.jl")
using .TimoshenkoBeam
export TimoshenkoBeamTransfun

include("./lu_solver.jl")
include("./base.jl")
include("./composition.jl")
include("./transformer.jl")
include("./parametric.jl")
include("./systems.jl")
include("./parametric_systems.jl")
include("./objective.jl")

function add_optim_parser(s)
  @add_arg_table! s begin
    "--store_objective_values"
    arg_type = Bool
    default = false
    "--show_trace"
    arg_type = Bool
    default = false
    "--allow_f_increases"
    arg_type = Bool
    default = false
    "--iterations"
    arg_type = Int
    default = 15000
    "--f_tol"
    arg_type = Float64
    default = 1e-8
    "--x_tol"
    arg_type = Float64
    default = 1e-8
    "--g_tol"
    arg_type = Float64
    default = 1e-8
  end

  return s
end

function add_default_options_parser(s)
  @add_arg_table! s begin
    "--reduced_order"
    arg_type = Int
    default = 20
    "--n_initial_frequency_samples"
    arg_type = Int
    default = 100
    "--refine_grid"
    arg_type = Bool
    default = true
    "--refine_tol"
    arg_type = Float64
    default = 0.1
  end
  return s
end

function default_optim_opts(;
  show_trace=false,
  store_trace=false,
  allow_f_increases=false,
  iterations=15000,
  f_tol=1e-8,
  x_tol=1e-8,
  g_tol=1e-8,
)
  optim_opts = Optim.Options(
    show_trace=show_trace,
    store_trace=store_trace,
    allow_f_increases=allow_f_increases,
    iterations=iterations,
    f_tol=f_tol,
    x_tol=x_tol,
    g_tol=g_tol,
  )
  return optim_opts
end

function get_err(given_tf, update_s)
  function errfun(s)
    return opnorm(given_tf(s) - update_s(s))
  end
end

function get_err(given_tf, update_s, update_p)
  function errfun(s)
    update_p(s[1])
    return opnorm(given_tf(s) - update_s(s[2]))
  end
end

function get_err(given_tf, update_funs::T) where {T<:Tuple}
  if length(update_funs) == 3
    return get_err(given_tf, update_funs[1], update_funs[2])
  elseif length(update_funs) == 2
    return get_err(given_tf, update_funs[1])
  else
    throw(ErrorException("update_funs has wrong length"))
  end
end

function get_values(trace)
  values = zeros(length(trace))
  for (i, t) in enumerate(trace)
    values[i] = t.value
  end
  return values
end

function sobmor_bisection(
  cfm,
  update_funs,
  mesh,
  targets,
  theta0;
  gamma_max=1.0,
  gamma_min=0.0,
  optim_opts=default_optim_opts(),
  given_tf=nothing,
  refine_grid=true,
  refine_tol=1e-2,
)
  if refine_grid
    errfun = get_err(given_tf, update_funs)
  end
  if optim_opts.store_trace == true
    @info "Storing objective values"
    loss_values = []
  end
  while (gamma_max - gamma_min) > (gamma_max + gamma_min) * 1e-2
    gamma = (gamma_max + gamma_min) / 2
    if !refine_grid
      targets = targets
    else
      refine!(mesh, errfun, gamma * refine_tol)
      targets = given_tf.(collect(mesh))
    end
    obj = l2_lift_objective(cfm, update_funs, collect(mesh), targets, gamma)
    loss = Optim.only_fg!((F, G, theta) -> obj(F, G, theta))
    tpre = time()
    res = Optim.optimize(loss, theta0, BFGS(), optim_opts)
    tpost = time()
    msg = @sprintf(
      "Gamma: %.2e, minimum: %.2e, iterations: %.2e, n_samples: %.2e, took: %.2es\n",
      gamma,
      res.minimum,
      res.iterations,
      length(mesh),
      tpost - tpre,
    )
    if optim_opts.store_trace == true
      push!(loss_values, get_values(res.trace))
    end
    @info "$(msg)"
    if res.minimum > 1e-12
      gamma_min = gamma
    else
      gamma_max = gamma
      theta0 = res.minimizer
    end
  end
  if optim_opts.store_trace == true
    for (i, vals) in enumerate(loss_values)
      writedlm("data/opt_vals_iter_$(i).txt", vals, ' ')
    end
  end
  return theta0
end

export get_split,
  partitioner,
  even_partition,
  FunMatrix,
  SVDInplace,
  l2_lift_objective,
  TriangularReshapes,
  linfnorm,
  linferr,
  sobmor_bisection,
  get_hatfun,
  weighted_sum,
  default_optim_opts
end
