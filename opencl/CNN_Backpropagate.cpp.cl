#ifndef CNN_BACKPROPAGATE_CPP_CL
#define CNN_BACKPROPAGATE_CPP_CL

// Note: Depends on CNN_Defines.hpp.cl (TYPE)

//===================================================================================================================//

// Computes ∂Cost/∂maxpool_input: routes gradient to max position (backward pass)
// nElements = C * outH * outW (same as forward output size)
kernel void calculate_dCost_dMaxpool(global TYPE* grads, global ulong* pool_indices,
                                     ulong grad_out_offset, // offset of pool output gradient (source)
                                     ulong pool_idx_offset, // offset into pool_indices for this layer
                                     ulong size // C * outH * outW
)
{
  size_t gid = get_global_id(0);

  if (gid >= size)
    return;

  TYPE dOutVal = grads[grad_out_offset + gid];
  ulong maxIdx = pool_indices[pool_idx_offset + gid];

  // Route gradient to the max position in the input gradient buffer
  // maxIdx is an absolute index into the actvs/grads buffer
  grads[maxIdx] += dOutVal;
}

//===================================================================================================================//

// Computes ∂Cost/∂avgpool_input: distributes gradient evenly across pooling window (backward pass)
// One work-item per output element (c, oh, ow)
// nElements = C * outH * outW
kernel void calculate_dCost_dAvgpool(global TYPE* grads,
                                     ulong grad_in_offset, // offset of pool input gradient (destination)
                                     ulong grad_out_offset, // offset of pool output gradient (source)
                                     ulong channels, ulong inputH, ulong inputW, ulong poolH, ulong poolW,
                                     ulong strideY, ulong strideX, ulong outH, ulong outW)
{
  size_t gid = get_global_id(0);

  ulong totalOut = channels * outH * outW;

  if (gid >= totalOut)
    return;

  ulong c = gid / (outH * outW);
  ulong rem = gid % (outH * outW);
  ulong oh = rem / outW;
  ulong ow = rem % outW;

  TYPE dOutVal = grads[grad_out_offset + gid];
  TYPE distributed = dOutVal / (TYPE)(poolH * poolW);

  for (ulong ph = 0; ph < poolH; ph++) {
    for (ulong pw = 0; pw < poolW; pw++) {
      ulong ih = oh * strideY + ph;
      ulong iw = ow * strideX + pw;

      grads[grad_in_offset + c * inputH * inputW + ih * inputW + iw] += distributed;
    }
  }
}

//===================================================================================================================//

// Computes ∂Cost/∂z = ∂Cost/∂a · (z > 0) (backward pass)
// One work-item per element
kernel void calculate_dCost_dRelu(global TYPE* grads, global TYPE* actvs,
                                  ulong grad_in_offset, // offset to write input gradient
                                  ulong grad_out_offset, // offset to read output gradient
                                  ulong actv_in_offset, // offset of forward input activations
                                  ulong size)
{
  size_t gid = get_global_id(0);

  if (gid >= size)
    return;

  TYPE actv = actvs[actv_in_offset + gid];
  TYPE dOut = grads[grad_out_offset + gid];

  grads[grad_in_offset + gid] = (actv > (TYPE)0) ? dOut : (TYPE)0;
}

//===================================================================================================================//

// Computes ∂Cost/∂b for convolution biases (backward pass)
// One work-group per filter. Work-items split the output positions, then tree-reduce.
// Global work size = numFilters * localWorkSize. Local work size must be power of 2.
kernel void calculate_dCost_dBiases(global TYPE* grads, global TYPE* dBiases, ulong grad_out_offset, ulong dbias_offset,
                                    ulong numFilters, ulong outH, ulong outW)
{
  local TYPE partials[256];

  size_t groupId = get_group_id(0); // filter index
  size_t lid = get_local_id(0);
  size_t localSize = get_local_size(0);

  ulong f = groupId;
  ulong totalOutPositions = outH * outW;
  ulong gradFBase = grad_out_offset + f * totalOutPositions;

  TYPE sum = (TYPE)0;

  for (ulong pos = lid; pos < totalOutPositions; pos += localSize) {
    sum += grads[gradFBase + pos];
  }

  partials[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Tree reduction
  for (ulong stride = localSize / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      partials[lid] += partials[lid + stride];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    dBiases[dbias_offset + f] = partials[0];
  }
}

//===================================================================================================================//

#endif // CNN_BACKPROPAGATE_CPP_CL
