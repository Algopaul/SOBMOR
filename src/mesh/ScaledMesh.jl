mutable struct ScaledMesh{TM,TL,TV}
  mesh::TM
  axis_log::TL
  axis_imag::TL
  abs_points::TV
end

Scaled1DMesh = ScaledMesh{TM} where {TM<:Mesh1D}

function ScaledMesh(
  m::Mesh;
  axis_log=zeros(Bool, length(m.points[1])),
  axis_imag=zeros(Bool, length(m.points[1])),
)
  Tc = eltype(m.points[1])
  Tc = any(axis_log) ? float(Tc) : Tc
  Tc = any(axis_imag) ? complex(Tc) : Tc
  abs_points = Vector{Vector{Tc}}(undef, 0)
  sm = ScaledMesh(m, axis_log, axis_imag, abs_points)
  update_abs_points!(sm)
  return sm
end

function ScaledMesh(m::Mesh1D; axis_log=false, axis_imag=false)
  Tc = typeof(m.points[1])
  Tc = any(axis_log) ? float(Tc) : Tc
  Tc = any(axis_imag) ? complex(Tc) : Tc
  abs_points = Vector{Tc}(undef, 0)
  sm = ScaledMesh(m, axis_log, axis_imag, abs_points)
  update_abs_points!(sm)
  return sm
end

n_edges(sm::ScaledMesh) = n_edges(sm.mesh)
Base.length(sm::ScaledMesh) = length(sm.mesh)
Base.getindex(m::ScaledMesh, I...) = getindex(m.abs_points, I...)
function Base.iterate(m::AdaptiveMesh.ScaledMesh, i=1)
  return length(m) >= i ? (getindex(m, i), i + 1) : nothing
end

function update_abs_points!(sm::ScaledMesh)
  n_points = length(sm.mesh.points)
  sm.abs_points = Vector{eltype(sm.mesh.points)}(undef, n_points)
  for i = 1:n_points
    sm.abs_points[i] = imag_if_necessary(expcoords(sm.mesh.points[i], sm), sm)
  end
  return nothing
end

function imag_if_necessary(v::AbstractVector, mesh)
  vv = complex(deepcopy(v))
  for i = 1:length(vv)
    if mesh.axis_imag[i]
      vv[i] = im * vv[i]
    end
  end
  return vv
end

function imag_if_necessary(v::Number, mesh)
  if mesh.axis_imag == true
    return im * v
  else
    return v
  end
end

function expcoords(p::AbstractVector, mesh)
  pabs = deepcopy(p)
  for i = 1:length(p)
    if mesh.axis_log[i]
      pabs[i] = exp10(pabs[i])
    end
  end
  return pabs
end

function expcoords(p::Number, mesh)
  if mesh.axis_log
    return exp10(p)
  else
    return p
  end
end

function update_mesh(sm::ScaledMesh)
  update_mesh(sm.mesh)
  update_abs_points!(sm)
end

expcoords(i::Int, sm::Mesh{TD}) where {TD<:AbstractVector{TV} where {TV<:AbstractVector}} =
  expcoords(sm.mesh.points[i], sm)

export ScaledMesh
