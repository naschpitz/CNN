#ifndef CNN_GLOBALAVGPOOL_CPP_CL
#define CNN_GLOBALAVGPOOL_CPP_CL

//===================================================================================================================//
// CNN_GlobalAvgPool.cpp.cl
//
// Global Average Pooling kernels.
// Forward: average each channel's spatial dims to produce (C, 1, 1).
// Backward: distribute gradient evenly across spatial positions.
//===================================================================================================================//

//===================================================================================================================//
//-- Forward pass --//
//===================================================================================================================//

// One work-group per channel. Uses tree reduction to compute the spatial mean.
// Writes the result to actvs[outOffset + c].
kernel void gap_propagate(global TYPE* actvs, ulong inOffset, ulong outOffset, ulong C, ulong H, ulong W)
{
  local TYPE partials[256];

  size_t c = get_group_id(0);
  size_t lid = get_local_id(0);
  size_t localSize = get_local_size(0);

  ulong spatialSize = H * W;
  ulong base = inOffset + c * spatialSize;

  TYPE sum = (TYPE)0;

  for (ulong s = lid; s < spatialSize; s += localSize)
    sum += actvs[base + s];

  partials[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (ulong stride = localSize / 2; stride > 0; stride >>= 1) {
    if (lid < stride)
      partials[lid] += partials[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0)
    actvs[outOffset + c] = partials[0] / (TYPE)spatialSize;
}

//===================================================================================================================//
//-- Backward pass --//
//===================================================================================================================//

// One work-item per element. Distributes the per-channel gradient evenly.
// nElements = C * H * W (the input spatial size).
kernel void gap_backpropagate(global TYPE* grads, ulong gradInOffset, ulong gradOutOffset, ulong C, ulong H, ulong W)
{
  size_t gid = get_global_id(0);
  ulong spatialSize = H * W;
  ulong totalSize = C * spatialSize;

  if (gid >= totalSize)
    return;

  ulong c = gid / spatialSize;
  TYPE invSpatial = (TYPE)1 / (TYPE)spatialSize;

  grads[gradInOffset + gid] = grads[gradOutOffset + c] * invSpatial;
}

//===================================================================================================================//

#endif // CNN_GLOBALAVGPOOL_CPP_CL
