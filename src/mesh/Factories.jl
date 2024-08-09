function MeshGrid2d(Ps1::AbstractVector{V}, Ps2::AbstractVector{V}) where {V<:Number}
  np1 = length(Ps1)
  np2 = length(Ps2)
  points = Vector{Vector{V}}(undef, np1 * np2)
  ij = 0
  for i = 1:np1
    for j = 1:np2
      ij += 1
      points[ij] = [Ps1[i], Ps2[j]]
    end
  end
  m = Mesh(points)
  update_edge_orientations!(m)
  update_mesh(m)
  return m
end

function MeshGrid3d(
  P1::AbstractVector{TV},
  P2::AbstractVector{TV},
  P3::AbstractVector{TV},
) where {TV<:Number}
  Ns = length.([P1, P2, P3])
  points = Vector{Vector{TV}}(undef, prod(Ns))
  ij = 0
  for iN = 1:Ns[1]
    for jN = 1:Ns[2]
      for zN = 1:Ns[3]
        ij += 1
        points[ij] = [P1[iN], P2[jN], P3[zN]]
      end
    end
  end
  m = Mesh(points)
  update_edge_orientations!(m)
  update_mesh(m)
  return m
end

export mesh1d, MeshGrid2d, MeshGrid3d
