#ifndef CNN_GLOBALDUALPOOL_CPP_CL
#define CNN_GLOBALDUALPOOL_CPP_CL

//===================================================================================================================//
// CNN_GlobalDualPool.cpp.cl
//
// Global Dual Pooling kernels (avg + max concatenated).
// Forward: for each channel, compute avg and max of spatial dims.
//          Output: (2C, 1, 1) — first C = avg, next C = max.
// Backward: avg gradient distributed uniformly; max gradient flows to max element only.
//===================================================================================================================//

//===================================================================================================================//
//-- Forward pass --//
//===================================================================================================================//

// One work-group per channel. Uses tree reduction to compute both avg and max.
// Writes avg to actvs[outOffset + c], max to actvs[outOffset + C + c].
kernel void gdp_propagate(global TYPE* actvs, ulong inOffset, ulong outOffset, ulong C, ulong H, ulong W)
{
  local TYPE partialSum[256];
  local TYPE partialMax[256];

  size_t c = get_group_id(0);
  size_t lid = get_local_id(0);
  size_t localSize = get_local_size(0);

  ulong spatialSize = H * W;
  ulong base = inOffset + c * spatialSize;

  TYPE sum = (TYPE)0;
  TYPE maxVal = -INFINITY;

  for (ulong s = lid; s < spatialSize; s += localSize) {
    TYPE val = actvs[base + s];
    sum += val;

    if (val > maxVal)
      maxVal = val;
  }

  partialSum[lid] = sum;
  partialMax[lid] = maxVal;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (ulong stride = localSize / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      partialSum[lid] += partialSum[lid + stride];

      if (partialMax[lid + stride] > partialMax[lid])
        partialMax[lid] = partialMax[lid + stride];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    actvs[outOffset + c] = partialSum[0] / (TYPE)spatialSize;
    actvs[outOffset + C + c] = partialMax[0];
  }
}

//===================================================================================================================//
//-- Backward pass --//
//===================================================================================================================//

// One work-item per spatial element. Distributes avg gradient uniformly
// and adds max gradient to the first max element only.
// Input gradient at gradOutOffset has 2C values: first C = dAvg, next C = dMax.
// actvs buffer is needed to find the max index per channel.
kernel void gdp_backpropagate(global TYPE* grads, global const TYPE* actvs, ulong gradInOffset, ulong gradOutOffset,
                              ulong actvsInOffset, ulong C, ulong H, ulong W)
{
  size_t gid = get_global_id(0);
  ulong spatialSize = H * W;
  ulong totalSize = C * spatialSize;

  if (gid >= totalSize)
    return;

  ulong c = gid / spatialSize;
  ulong s = gid % spatialSize;
  ulong base = actvsInOffset + c * spatialSize;

  TYPE dAvg = grads[gradOutOffset + c];
  TYPE dMax = grads[gradOutOffset + C + c];
  TYPE invSpatial = (TYPE)1 / (TYPE)spatialSize;

  // Avg gradient: uniform distribution
  TYPE grad = dAvg * invSpatial;

  // Max gradient: find if this element is the first max
  TYPE myVal = actvs[base + s];
  TYPE maxVal = -INFINITY;
  ulong maxIdx = 0;

  for (ulong i = 0; i < spatialSize; i++) {
    TYPE val = actvs[base + i];

    if (val > maxVal) {
      maxVal = val;
      maxIdx = i;
    }
  }

  if (s == maxIdx)
    grad += dMax;

  grads[gradInOffset + gid] = grad;
}

//===================================================================================================================//

#endif // CNN_GLOBALDUALPOOL_CPP_CL

