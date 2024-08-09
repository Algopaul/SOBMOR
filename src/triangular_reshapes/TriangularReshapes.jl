module TriangularReshapes

using LoopVectorization

"""
    vector_to_lower_triang!(M::AbstractMatrix{T}, v::AbstractVector{T}) where {T}

Overwrites the lower triangular part of `M` with the values of `v`.
# Arguments
- `M`: The matrix to be overwritten.
- `v`: The vector to be used to overwrite the lower triangular part of `M`.
"""
function vector_to_lower_triang!(M::AbstractMatrix{T}, v::AbstractVector{T}) where {T}
  n = size(M, 1)
  @assert size(M, 2) == n
  @assert length(v) >= n * (n + 1) / 2
  st = 1
  l = n
  Base.require_one_based_indexing(M, v)
  @inbounds @simd for i = 1:n
    @turbo warn_check_args = false for j = 0:(n-i)
      M[j+i, i] = v[st+j]
    end
    st += l
    l -= 1
  end
  return nothing
end

"""
    vector_to_lower_triang(v::AbstractVector)

Creates a matrix `M`, whose lower triangular part is filled with the values of v.
# Arguments
- `v`: The vector to be used to overwrite the lower triangular part of `M`.
# Outputs
- `M`: The matrix whose lower triangular part is filled with the values of `v`.
"""
function vector_to_lower_triang(v::AbstractVector{T}) where {T}
  n = length(v)
  nM = Int(-0.5 + sqrt(0.25 + 2 * n))
  M = zeros(T, nM, nM)
  vector_to_lower_triang!(M, v)
  return M
end

"""
    lower_triang_to_vector!(v::AbstractVector{T}, M::AbstractMatrix{T}) where {T}

Overwrites the first `n * (n + 1) / 2` elements of `v` with the values of the lower triangular part of `M`.
# Arguments
- `v`: The vector to be overwritten.
- `M`: The matrix whose lower triangular part is used to overwrite the first `n * (n + 1) / 2` elements of `v`.
"""
function lower_triang_to_vector!(v::AbstractVector{T}, M::AbstractMatrix{T}) where {T}
  n = size(M, 1)
  @assert size(M, 2) == n
  @assert length(v) >= n * (n + 1) / 2
  st = 1
  l = n
  Base.require_one_based_indexing(M, v)
  @inbounds @simd for i = 1:n
    @turbo warn_check_args = false for j = 0:(n-i)
      v[st+j] = M[j+i, i]
    end
    st += l
    l -= 1
  end
  return nothing
end

"""
    lower_triang_to_vector(M::AbstractMatrix{T}) where {T}

Returns a vector containing the values of the lower triangular part of `M`.
# Arguments
- `M`: The matrix whose lower triangular part is used to create the vector.
# Outputs
- `v`: The vector containing the values of the lower triangular part of `M`.
"""
function lower_triang_to_vector(M::AbstractMatrix{T}) where {T}
  n = size(M, 1)
  v = Vector{T}(undef, n * (n + 1) ÷ 2)
  lower_triang_to_vector!(v, M)
  return v
end

for c1 in (true, false)
  for c2 in (true, false)
    for makereal in (true, false)
      c1name = c1 ? "_c1" : ""
      c1fun = c1 ? conj : identity
      c2name = c2 ? "_c2" : ""
      c2fun = c2 ? conj : identity
      realname = makereal ? "_real" : ""
      realfun = makereal ? real : identity
      ltv_name = Symbol("ltv" * c1name * c2name * realname * "!")
      ltv_name_sum = Symbol("ltv" * c1name * c2name * realname * "_sum!")
      vtype = makereal ? Real : Number
      @eval begin
        function $ltv_name(
          v::AbstractVector{T1},
          v1::AbstractVector,
          v2::AbstractVector,
        ) where {T1<:$vtype}
          Base.require_one_based_indexing(v, v1, v2)
          n = length(v1)
          l = n
          st = 1
          @inbounds for i = 1:n
            v2i = v2[i]
            v2i = $c2fun(v2i)
            @simd for j = i:n
              v1j = $c1fun(v1[j])
              v1jv2i = $realfun(v1j * v2i)
              v[j-1+st] = v1jv2i
            end
            st += l - 1
            l -= 1
          end
          return nothing
        end
        function $ltv_name_sum(
          v::AbstractVector{T1},
          v1::AbstractVector,
          v2::AbstractVector,
        ) where {T1<:$vtype}
          Base.require_one_based_indexing(v, v1, v2)
          n = length(v1)
          l = n
          st = 1
          @inbounds for i = 1:n
            v2i = v2[i]
            v2i = $c2fun(v2i)
            @simd for j = i:n
              v1j = $c1fun(v1[j])
              v1jv2i = $realfun(v1j * v2i)
              v[j-1+st] += v1jv2i
            end
            st += l - 1
            l -= 1
          end
          return nothing
        end
      end
    end
  end
end

"""
    lower_triang_to_vector!(v, v1, v2, c1, c2, adding)

Overwrites the first `n * (n + 1) / 2` elements of v with the values of the lower triangular part of `v1*transpose(v2)`.
# Arguments
- `v`: The vector to be overwritten.
- `v1, v2`: Vectors to form the rank-one-matrix `v1*v2'`
- `c1, c2`: Booleans to indicate whether the vectors `v1` or `v2` should be conjugated.
- `adding`: Boolean to indicate whether the output should be added to `v` or if `v` should be overwritten.
"""
function lower_triang_to_vector!(
  v::AbstractVector,
  v1::AbstractVector,
  v2::AbstractVector;
  c1=false::Bool,
  c2=false::Bool,
  adding=false::Bool,
)
  n = length(v1)
  @assert length(v2) == n
  @assert length(v) >= n * (n + 1) ÷ 2
  if c1
    if c2
      if adding
        ltv_c1_c2_sum!(v, v1, v2)
      else
        ltv_c1_c2!(v, v1, v2)
      end
    else
      if adding
        ltv_c1_sum!(v, v1, v2)
      else
        ltv_c1!(v, v1, v2)
      end
    end
  else
    if c2
      if adding
        ltv_c2_sum!(v, v1, v2)
      else
        ltv_c2!(v, v1, v2)
      end
    else
      if adding
        ltv_sum!(v, v1, v2)
      else
        ltv!(v, v1, v2)
      end
    end
  end
  return nothing
end

function lower_triang_to_vector!(
  v::AbstractVector{T},
  v1::AbstractVector,
  v2::AbstractVector;
  c1=false::Bool,
  c2=false::Bool,
  adding=false::Bool,
) where {T<:Real}
  n = length(v1)
  @assert length(v2) == n
  @assert length(v) >= n * (n + 1) ÷ 2
  if c1
    if c2
      if adding
        ltv_c1_c2_real_sum!(v, v1, v2)
      else
        ltv_c1_c2_real!(v, v1, v2)
      end
    else
      if adding
        ltv_c1_real_sum!(v, v1, v2)
      else
        ltv_c1_real!(v, v1, v2)
      end
    end
  else
    if c2
      if adding
        ltv_c2_real_sum!(v, v1, v2)
      else
        ltv_c2_real!(v, v1, v2)
      end
    else
      if adding
        ltv_real_sum!(v, v1, v2)
      else
        ltv_real!(v, v1, v2)
      end
    end
  end
  return nothing
end

include("./StrictlyLowerTriang.jl")

export vector_to_lower_triang!,
  lower_triang_to_vector!, lower_triang_to_vector, vector_to_lower_triang

end # module TriangularReshapes
