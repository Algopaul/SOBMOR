function refine!(mesh, fun, tol)
  i_start = 1
  counter = 1
  while i_start <= n_edges(mesh)
    i_start = refine_i!(mesh, fun, tol, i_start)
  end
  return nothing
end

function refine_i!(mesh, fun, tol, i_start=1)
  n_edges_0 = n_edges(mesh)
  inds = get_critical_edges(mesh, fun, tol, i_start)
  for idx in sort(inds, rev=true)
    split_edge(idx, mesh)
  end
  n_edges_m = n_edges(mesh)
  update_mesh(mesh)
  new_i_start = n_edges_0 - (n_edges_m - n_edges_0) + 1
  return new_i_start
end

function refine!(mesh::T, fun, tol) where {T<:Union{Scaled1DMesh,Mesh1D}}
  while refine_i!(mesh, fun, tol)
  end
  return nothing
end

get_mesh_points(mesh::Mesh) = mesh.points
get_mesh_points(mesh::ScaledMesh) = mesh.mesh.points
function set_mesh_points(mesh::Mesh, points)
  mesh.points = points
end
function set_mesh_points(mesh::ScaledMesh, points)
  mesh.mesh.points = points
end

function refine_i!(mesh::T, fun, tol) where {T<:Union{Scaled1DMesh,Mesh1D}}
  E = [0, 0]
  points = get_mesh_points(mesh)
  new_samples = Vector{eltype(points)}(undef, 0)
  for (i, j) in enumerate(2:length(points))
    E .= (i, j)
    if check_edge([i, j], fun, tol, mesh)
      push!(new_samples, (points[i] + points[j]) / 2)
    end
  end
  if length(new_samples) > 0
    all_points = sort(vcat(points, new_samples))
    set_mesh_points(mesh, all_points)
    update_mesh(mesh)
    return true
  else
    update_mesh(mesh)
    return false
  end
end

function get_critical_edges(m, fun, tol, i_start=1)
  crit_edges = Vector{Int}(undef, 0)
  for i = i_start:n_edges(m)
    if check_edge(get_edge(m, i), fun, tol, m)
      push!(crit_edges, i)
    end
  end
  return crit_edges
end

get_edge(m::Mesh, i::Int) = m.edges[i]
get_edge(m::ScaledMesh, i::Int) = m.mesh.edges[i]

function split_edge(i::Int, mesh::ScaledMesh)
  split_edge(i, mesh.mesh)
end

function split_edge(i::Int, mesh::Mesh)
  edge = mesh.edges[i]
  p1, p2, pmp = endpoints(edge, mesh) |> with_mean
  popat!(mesh.edges, i)
  k = popat!(mesh.edge_orientations, i)
  push!(mesh.points, pmp)
  push!(mesh.edges, [edge[1], length(mesh.points)])
  push!(mesh.edges, [length(mesh.points), edge[2]])
  push!(mesh.edge_orientations, k)
  push!(mesh.edge_orientations, k)
end

function check_edge(edge, fun, tol, mesh::ScaledMesh, absco=p -> abscoords(p, mesh))
  check_edge(edge, fun, tol, mesh.mesh, absco)
end

function check_edge(edge, fun, tol, mesh::Mesh, absco=p -> abscoords(p, mesh))
  s0, s1, smp = endpoints(edge, mesh) |> with_mean
  ωs0, ωs1, ωsmp = absco.([s0, s1, smp])
  fs0, fs1, fsmp = fun(ωs0), fun(ωs1), fun(ωsmp)
  d1 = abs(fsmp - fs0) / norm(ωs0 - ωsmp)
  d2 = abs(fs1 - fsmp) / norm(ωs1 - ωsmp)
  return inner_point_required(max(d1, d2), ωs0, ωs1, fs0, fs1, tol, mesh)
end

function abscoords(p, mesh::Mesh)
  return p
end

function abscoords(p, sm::ScaledMesh)
  return imag_if_necessary(expcoords(p, sm), sm)
end

function with_mean(A)
  a, b = A
  return with_mean(a, b)
end

function with_mean(a, b)
  return a, b, (a + b) / 2
end

function inner_point_required(deriv_bound, x1, x2, f1, f2, tol, mesh)
  γ = max(f1, f2)
  return deriv_bound * dist(x2, x1, mesh) >= 2 * (γ + tol) - (f1 + f2)
end
