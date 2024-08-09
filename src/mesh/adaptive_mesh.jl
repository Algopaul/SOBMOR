module AdaptiveMesh

using LinearAlgebra

mutable struct Mesh{TP,TE,TI}
  points::Vector{TP}
  edges::TE
  edge_orientations::TI
  function Mesh(
    points::AbstractVector{TV},
    edges=Vector{Vector{Int}}(undef, 0),
    edge_orientations=Vector{Int}(undef, 0),
  ) where {TV}
    et = typeof(edges)
    eot = typeof(edge_orientations)
    mesh = new{TV,et,eot}(Vector(points), edges, edge_orientations)
    update_edge_orientations!(mesh)
    return mesh
  end
end

Mesh1D = Mesh{TP} where {TP<:Number}

function update_mesh(mesh::Mesh, i_start=1)
  n_newpoints = update_mesh_i(mesh, i_start)
  return n_newpoints
end

function update_mesh(::Mesh{TP}, i_start=1) where {TP<:Number}
  return nothing
end

function update_mesh_i(mesh, i_start=1)
  n_dims = length(mesh.points[1])
  counter = 0
  for i = i_start:length(mesh.points)
    for dim = 1:n_dims
      for dir in [1, -1]
        counter += create_new_edge_if_needed(i, mesh, dim, dir)
      end
    end
  end
  return counter
end

function create_new_edge_if_needed(i::Int, mesh, dim, dir)
  j = next_point(i, mesh, dim, dir)
  if j !== nothing
    if !edge_between(mesh.points[i], mesh.points[j], mesh)
      push!(mesh.edges, [i, j])
      push!(mesh.edge_orientations, get_dim(mesh.points[i], mesh.points[j]))
      return 1
    end
  end
  return 0
end

function update_edge_orientations!(mesh)
  n = length(mesh.edges)
  mesh.edge_orientations = Vector{Int}(undef, n)
  for i = 1:n
    p1, p2 = endpoints(mesh.edges[i], mesh)
    mesh.edge_orientations[i] = get_dim(p1, p2)
  end
end

function next_point(point, mesh, dim, dir)
  cands = findall(
    s -> (equal_coords_axis(s, point, dim) && correct_dir(point, s, dim, dir)),
    mesh.points,
  )
  if length(cands) == 0
    return nothing
  end
  val, i = findmin([dist(s, point, mesh) for s in mesh.points[cands]])
  i = cands[i]
end

function next_point(i::Int, mesh, dim, dir)
  next_point(mesh.points[i], mesh, dim, dir)
end

function edge_between(p1, p2, mesh)
  if check_if_edge_exists(p1, p2, mesh)
    return true
  else
    ch_dim = findfirst(s -> (p1-p2)[s] != 0, 1:length(p1))
    d1 = min(p1[ch_dim], p2[ch_dim])
    d2 = max(p1[ch_dim], p2[ch_dim])
    for (i, edge) in enumerate(mesh.edges)
      p3, p4 = endpoints(edge, mesh)
      dim = mesh.edge_orientations[i]
      if check_if_between(p1, p2, p3, p4, ch_dim, d1, d2, dim)
        return true
      end
    end
    return false
  end
end

function check_if_edge_exists(p1, p2, mesh)
  for edge in mesh.edges
    p3, p4 = endpoints(edge, mesh)
    if (p3, p4) == (p1, p2) || (p4, p3) == (p1, p2)
      return true
    end
  end
  return false
end

function check_if_between(
  p1,
  p2,
  p3,
  p4,
  ch_dim=findfirst(s -> (p1-p2)[s] != 0, 1:length(p1)),
  d1=min(p1[ch_dim], p2[ch_dim]),
  d2=max(p1[ch_dim], p2[ch_dim]),
  dim=get_dim(p3, p4),
)
  if dim == ch_dim
    return false
  end
  if !(d1 < p3[ch_dim] < d2)
    return false
  end
  if !(d1 < p4[ch_dim] < d2)
    return false
  end
  for i in setdiff(1:length(p1), [dim, ch_dim])
    if p2[i] != p4[i]
      return false
    end
  end
  d1 = min(p3[dim], p4[dim])
  d2 = max(p3[dim], p4[dim])
  if d1 < p1[dim] < d2
    return true
  else
    return false
  end
end

edge_between(i::Int, j::Int, mesh) = edge_between(mesh.points[i], mesh.points[j], mesh)

function endpoints(edge, mesh)
  return mesh.points[edge[1]], mesh.points[edge[2]]
end

function dist(p1, p2, mesh)
  return norm(p1 - p2)
end

dist(i::Int, j::Int, mesh) = dist(mesh.points[i], mesh.points[j], mesh)

function dist(p1, p2)
  return norm(p1 - p2)
end

function get_dim(p1, p2)
  @inbounds for i = 1:length(p1)
    if p1[i] != p2[i]
      return i
    end
  end
  return length(p1)
end

function hamming_norm(v::AbstractVector{T}) where {T}
  k = 0
  @inbounds for i in eachindex(v)
    if v[i] != zero(T)
      k += 1
    end
  end
  return k
end

function correct_dir(p1, p2, dim, dir)
  if dir * p1[dim] < dir * p2[dim]
    return true
  else
    return false
  end
end

function equal_coords_axis(p1, p2, dim::Int)
  for i = 1:length(p1)
    if i != dim
      if p1[i] != p2[i]
        return false
      end
    end
  end
  return true
end

function Base.length(m::Mesh)
  return length(m.points)
end

function n_edges(m::Mesh)
  return length(m.edges)
end

Base.getindex(m::Mesh, I...) = getindex(m.points, I...)
Base.iterate(m::AdaptiveMesh.Mesh, i=1) = length(m) >= i ? (getindex(m, i), i + 1) : nothing

include("./Factories.jl")
include("./ScaledMesh.jl")
include("./Refinement.jl")

export Mesh, update_mesh, refine!, refine_i!

end
