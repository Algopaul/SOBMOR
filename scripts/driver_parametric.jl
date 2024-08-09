using Pkg
using DelimitedFiles
Pkg.activate(".")
using ArgParse
using Random
using SOBMOR
using Optim
using Printf
using LinearAlgebra
using PortHamiltonianBenchmarkSystems

@info "System Configuration:"
@info "BLAS nthreads: $(BLAS.get_num_threads())"

s = ArgParseSettings()
s = SOBMOR.add_optim_parser(s)
s = SOBMOR.add_default_options_parser(s)

@add_arg_table! s begin
  "--outname"
  arg_type = String
  default = "timoshenko_hinf_errors"
end

args = parse_args(ARGS, s)

const r = args["reduced_order"]

pSamp = range(0.4, 2.4, length=20)
samples = SOBMOR.ScaledMesh(
  SOBMOR.MeshGrid2d(pSamp, range(-8, 6, length=args["n_initial_frequency_samples"])),
  axis_imag=[false, true],
  axis_log=[false, true],
);

r = args["reduced_order"]
cfm, update_s, update_p, update_model, theta0, matrices =
  get_stable_parametric(r, 1, 1, 6, 0.4, 2.4);

update_model(theta0)

@info "Initialized initial reduced order system with r=$r"

H = TimoshenkoBeamTransfun
targets = H.(collect(samples))

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
  (update_s, update_p, update_model),
  samples,
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
# Evaluate hinfnorm on fine paramter grid
p_grid = range(0.4, stop=2.4, length=200)

function beam_hinferr(p)
  update_p(p)
  _, _, _, Br, Cr, FTr, _, JmRQr, _, _ = matrices
  Ar = real(JmRQr.target_matrix)
  Dr = real(FTr.target_matrix)
  Br = real(Br.target_matrix)
  Cr = real(Cr.target_matrix)
  Er = Array{Float64}(I(size(Ar)[1]))
  Ef, Af, Bf, Cf = SOBMOR.TimoshenkoBeam.fem_beam(p)
  Ef = Matrix(Ef)
  Af = Matrix(Af)
  Bf = Matrix(Bf)
  Cf = Matrix(Cf)
  normval, _ = linferr(Af, Bf, Cf, zeros(1, 1), Ar, Br, Cr, Dr, E1=Ef, E2=Er)
  return normval[1]
end

hinferrs = beam_hinferr.(p_grid)
println(hinferrs)
error_array = [p_grid hinferrs]

writedlm("data/$(args["outname"]).txt", error_array, ' ')
