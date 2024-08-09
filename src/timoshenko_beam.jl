module TimoshenkoBeam

using LinearAlgebra, SparseArrays, Memoize

"""
Constructs the system matrices for the TimoshenkoBeam model. The code is copied
from http://mediatum.ub.tum.de/doc/1072355/document.pdf and only modified to
fit the julia syntax.
"""
function fem_beam(L)
  #
  # (input) L: total beam length (x) [m]
  N = 20 # number of elements [1]
  t = 0.01 # t: beam thickness (y) [m]
  h = 0.01 # h: beam height (z) [m]
  rho = 7850 # density of steel [kg/m^3]
  m = L * t * h * rho # m: total beam mass [kg]
  E = 210e9 # E: Young's modulus of steel [N/m^2]
  nu = 3 / 10 # nu: Poisson's ratio
  d1 = 8e-6
  d2 = 8 # d1, d2: dampening ratio (D = d1*K + d2*M)
  #
  # Physical Quantities
  G = E / 2 / (1 + nu) # G: Shear modulus [N/m^2]
  l = L / N # l: beam element length
  A = t * h  # beam area [m^2]
  ASy = 5 / 6 * A
  ASz = 5 / 6 * A # effective area of shear
  Iy = 1 / 12 * h^3 * t
  Iz = 1 / 12 * t^3 * h # second moments of area [m^4]
  Ip = 1 / 12 * t * h * (h^2 + t^2) # polar moment of inertia [m^4]
  It = minimum([h t])^3 * maximum([h t]) / 7 # torsion constant [m^4]
  Py = 12 * E * Iz / (G * ASy * l^2)
  Pz = 12 * E * Iy / (G * ASz * l^2) # Phi
  M11 = zeros(6, 6)
  M11[1, 1] = 1 / 3
  M11[2, 2] = 13 / 35 + 6 * Iz / (5 * A * l^2)
  M11[3, 3] = 13 / 35 + 6 * Iy / (5 * A * l^2)
  M11[4, 4] = Ip / (3 * A)
  M11[5, 5] = l^2 / 105 + 2 * Iy / (15 * A)
  M11[6, 6] = l^2 / 105 + 2 * Iz / (15 * A)
  M11[6, 2] = 11 * l / 210 + Iz / (10 * A * l)
  M11[2, 6] = M11[6, 2]
  M11[5, 3] = -11 * l / 210 - Iy / (10 * A * l)
  M11[3, 5] = M11[5, 3]
  M22 = -M11 + 2 * diagm(diag(M11))
  M21 = zeros(6, 6)
  M21[1, 1] = 1 / 6
  M21[2, 2] = 9 / 70 - 6 * Iz / (5 * A * l^2)
  M21[3, 3] = 9 / 70 - 6 * Iy / (5 * A * l^2)
  M21[4, 4] = Ip / (6 * A)
  M21[5, 5] = -l^2 / 140 - Iy / (30 * A)
  M21[6, 6] = -l^2 / 140 - Iz / (30 * A)
  M21[6, 2] = -13 * l / 420 + Iz / (10 * A * l)
  M21[2, 6] = -M21[6, 2]
  M21[5, 3] = 13 * l / 420 - Iy / (10 * A * l)
  M21[3, 5] = -M21[5, 3]
  Me = m / N * [M11 M21'; M21 M22]
  K11 = zeros(6, 6)
  K11[1, 1] = E * A / l
  K11[2, 2] = 12 * E * Iz / (l^3 * (1 + Py))
  K11[3, 3] = 12 * E * Iy / (l^3 * (1 + Pz))
  K11[4, 4] = G * It / l
  K11[5, 5] = (4 + Pz) * E * Iy / (l * (1 + Pz))
  K11[6, 6] = (4 + Py) * E * Iz / (l * (1 + Py))
  K11[2, 6] = 6 * E * Iz / (l^2 * (1 + Py))
  K11[6, 2] = K11[2, 6]
  K11[3, 5] = -6 * E * Iy / (l^2 * (1 + Pz))
  K11[5, 3] = K11[3, 5]
  K22 = -K11 + 2 * diagm(diag(K11))
  K21 = K11 - 2 * diagm(diag(K11))
  K21[5, 5] = (2 - Pz) * E * Iy / (l * (1 + Pz))
  K21[6, 6] = (2 - Py) * E * Iz / (l * (1 + Py))
  K21[2, 6] = -K21[6, 2]
  K21[3, 5] = -K21[5, 3]
  Ke = [K11 K21'; K21 K22]
  # global mass and stiffness matrices: N*(6+1) dofs
  M = spzeros(6 * (N + 1), 6 * (N + 1))
  K = spzeros(6 * (N + 1), 6 * (N + 1))
  for i = 1:N
    a = 1+6*(i-1):6*(i+1)
    @views K[a, a] = K[a, a] + Ke
    @views M[a, a] = M[a, a] + Me
  end
  K = K[7:end, :]
  K = K[:, 7:end]
  M = M[7:end, :]
  M = M[:, 7:end]
  D = d1 * K + d2 * M
  B = zeros(6 * N, 1)
  C = zeros(1, 6 * N)
  B[N*6-3, 1] = -1 # input: beam tip, -z
  C[1, N*6-3] = 1 # output: beam tip, +z
  F = Array{Float64}(I, 6 * N, 6 * N)
  E = [F zeros(6 * N, 6 * N); zeros(6 * N, 6 * N) M]
  A = [zeros(6 * N, 6 * N) F; -K -D]
  B = vcat(zeros(6 * N, size(B, 2)), B)
  C = hcat(C, zeros(size(C, 1), 6 * N))
  return E, A, B, C
end

"""
Returns the transfer function value of the Timoshenko beam model in
http://mediatum.ub.tum.de/doc/1072355/document.pdf \\
Cite as Panzer, H; Hubele, J.; Eid, R.; Lohmann, B.: Generating a Parametric
Finite Element Model of a 3D Cantilever Timoshenko Beam Using Matlab, Technical
Reports on Automatic Control, vol. TRAC-4, Institute of Automatic Control, TUM,
2009.

# Arguments
- s::ComplexF64 : Laplace variable
- L::Float64 : beam length ∈ (0.8, 2.0)

# Example
TimoshenkoBeamTransfun(5.0im, 1.2)
"""
@memoize function TimoshenkoBeamTransfun(s, L)
  E, A, B, C = fem_beam(L)
  return C * ((s * E - A) \ B)
end

@memoize function TimoshenkoBeamTransfun(V)
  # @assert length(V) == 2
  # E, A, B, C = fem_beam(V[1])
  # return C * ((V[2] * E - A) \ B)
  return TimoshenkoBeamTransfun(V[2], V[1])
end

function get_TimoshenkoBeamTransfun(L)
  E, A, B, C = fem_beam(L)
  return s -> C * ((s * E - A) \ B)
end

export TimoshenkoBeamTransfun, fem_beam

end # module
