using Pkg
Pkg.activate(".")
using ArgParse
using Random
using SOBMOR
using Optim
using Printf
using LinearAlgebra
using PortHamiltonianBenchmarkSystems
using Memoize

s = ArgParseSettings()
s = SOBMOR.add_optim_parser(s)
s = SOBMOR.add_default_options_parser(s)

# Add specific options
@add_arg_table! s begin
  "--use_Q"
  arg_type = Bool
  required = false
  default = false
  help = "Whether to use a pH model with/out Q matrix"
  "--store_frequency_response"
  arg_type = Bool
  default = true
end

args = parse_args(ARGS, s)

const r = args["reduced_order"]

freqs = SOBMOR.ScaledMesh(
  SOBMOR.Mesh(range(-8, stop=6, length=args["n_initial_frequency_samples"])),
  axis_log=true,
  axis_imag=true,
)

id_init = zeros(div(r * (r + 1), 2));
TriangularReshapes.lower_triang_to_vector!(id_init, Array(I(r) * 1.0));
j_init = ones(div(r * (r - 1), 2))
b_init = ones(r * 2)

function get_initial_rom(args)
  if args["use_Q"]
    cfm, update_s, update_model, matrices = get_pH_system(r, 2)
    theta0 = [
      j_init + 1e-3 * randn(MersenneTwister(0), size(j_init)...)
      id_init + 1e-3 * randn(MersenneTwister(1), size(id_init)...)
      id_init + 1e-3 * randn(MersenneTwister(2), size(id_init)...)
      b_init + 1e-3 * randn(MersenneTwister(3), size(b_init)...)
    ]
  else
    cfm, update_s, update_model, matrices = get_pH_system_noQ(r, 2)
    theta0 = [
      j_init + 1e-3 * randn(MersenneTwister(0), size(j_init)...)
      id_init + 1e-3 * randn(MersenneTwister(1), size(id_init)...)
      b_init + 1e-3 * randn(MersenneTwister(2), size(b_init)...)
    ]
  end
  return cfm, update_s, update_model, matrices, theta0
end

cfm, update_s, update_model, matrices, theta0 = get_initial_rom(args)

@info "Initialized initial reduced order system with r=$r"

config = SingleMSDConfig("Gugercin")
J, R, Q, B = construct_system(config)

@memoize function H(s)
  return B' * Q * ((s * I - (J - R) * Q) \ B)
end

targets = H.(freqs)
@info "Loaded target transfer function and computed initial samples"

optim_opts = default_optim_opts(
  show_trace=args["show_trace"],
  store_trace=args["store_objective_values"],
  allow_f_increases=args["allow_f_increases"],
  iterations=args["iterations"],
  f_tol=args["f_tol"],
  x_tol=args["x_tol"],
  g_tol=args["g_tol"],
)

@info "Starting SOBMOR"
tpre_sobmor = time();
theta0 = sobmor_bisection(
  cfm,
  (update_s, update_model),
  freqs,
  targets,
  theta0;
  optim_opts=optim_opts,
  given_tf=H,
  refine_grid=args["refine_grid"],
  refine_tol=args["refine_tol"],
)
tpost_sobmor = time();
msg = @sprintf("SOBMOR finished. Took %.2e seconds", tpost_sobmor - tpre_sobmor)
@info "$(msg)"

@info "Start evaluation"
freqs = collect(freqs)
results = [deepcopy(update_s(freqs[1]))]
for omeg in freqs[2:end]
  push!(results, deepcopy(update_s(omeg)))
end

using Plots
plot(imag(freqs), norm.(results), xaxis=:log, yaxis=:log)
plot!(imag(freqs), norm.(H.(freqs)))
plot!(imag(freqs), norm.(H.(freqs) - results))

A_full = (J - R) * Q
B_full = B
C_full = B' * Q
D_full = zeros(2, 2)

update_model(theta0);
function getABCD_red(matrices)
  if args["use_Q"]
    Jr, Rr, Qr, Br = matrices
    A_red = (Jr.target_matrix - Rr.target_matrix) * Qr.target_matrix
    B_red = Br.target_matrix
    C_red = Br.target_matrix' * Qr.target_matrix
    D_red = zeros(2, 2)
  else
    Jr, Rr, Br = matrices
    A_red = (Jr.target_matrix - Rr.target_matrix)
    B_red = Br.target_matrix
    C_red = Br.target_matrix'
    D_red = zeros(2, 2)
  end
  return A_red, B_red, C_red, D_red
end

A_red, B_red, C_red, D_red = getABCD_red(matrices)
@info "Condition number of ROM.A is $(cond(A_red))"
@info "Number of parameters is $(length(theta0))"
@info "Computing H-infinity error of ROM using SLICOT-routine `ab13dd`"
normval, hinffreq = linferr(A_full, B_full, C_full, D_full, A_red, B_red, C_red, D_red)
@info "H-infinity error of ROM with order r=$r is $(@sprintf("%.4e", normval)) and is attained at $(@sprintf("%.4e", hinffreq))im"
