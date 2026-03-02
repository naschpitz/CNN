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

// Computes ∂Cost/∂W for convolution filters (backward pass)
// One work-group per filter element (f, c, kh, kw). Work-items within a group split the
// output positions, compute partial sums, then tree-reduce in local memory.
// Global work size = numFilterElems * localWorkSize. Local work size must be power of 2.
kernel void calculate_dCost_dFilters(global TYPE* grads, global TYPE* actvs, global TYPE* dFilters,
                                     ulong grad_out_offset, ulong actv_in_offset, ulong dfilter_offset, ulong inputC,
                                     ulong inputH, ulong inputW, ulong numFilters, ulong filterH, ulong filterW,
                                     ulong strideY, ulong strideX, ulong padY, ulong padX, ulong outH, ulong outW)
{
  local TYPE partials[256];

  size_t groupId = get_group_id(0); // filter element index
  size_t lid = get_local_id(0);
  size_t localSize = get_local_size(0);

  ulong filterElemIdx = groupId;

  // Decompose filterElemIdx into (f, c, kh, kw)
  ulong f = filterElemIdx / (inputC * filterH * filterW);
  ulong rem = filterElemIdx % (inputC * filterH * filterW);
  ulong c = rem / (filterH * filterW);
  ulong rem2 = rem % (filterH * filterW);
  ulong kh = rem2 / filterW;
  ulong kw = rem2 % filterW;

  // Precompute base offsets
  ulong gradFBase = grad_out_offset + f * outH * outW;
  ulong actvCBase = actv_in_offset + c * inputH * inputW;

  // Each work-item processes a strided subset of output positions
  ulong totalOutPositions = outH * outW;
  TYPE sum = (TYPE)0;

  for (ulong pos = lid; pos < totalOutPositions; pos += localSize) {
    ulong oh = pos / outW;
    ulong ow = pos % outW;

    long ih = (long)(oh * strideY + kh) - (long)padY;
    long iw = (long)(ow * strideX + kw) - (long)padX;

    if (ih >= 0 && ih < (long)inputH && iw >= 0 && iw < (long)inputW) {
      sum += grads[gradFBase + pos] * actvs[actvCBase + (ulong)ih * inputW + (ulong)iw];
    }
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
    dFilters[dfilter_offset + filterElemIdx] = partials[0];
  }
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

// Computes ∂Cost/∂input via transposed convolution (backward pass)
// One work-item per input element: (c, ih, iw)
// nElements = inputC * inputH * inputW
kernel void calculate_dCost_dInput(global TYPE* grads, global TYPE* filters,
                                   ulong grad_out_offset, // offset of conv output gradient
                                   ulong grad_in_offset, // offset to write input gradient
                                   ulong filter_offset, // offset of this layer's filters
                                   ulong inputC, ulong inputH, ulong inputW, ulong numFilters, ulong filterH,
                                   ulong filterW, ulong strideY, ulong strideX, ulong padY, ulong padX, ulong outH,
                                   ulong outW)
{
  size_t gid = get_global_id(0);

  ulong totalInput = inputC * inputH * inputW;

  if (gid >= totalInput)
    return;

  // Decompose gid into (c, ih, iw)
  ulong c = gid / (inputH * inputW);
  ulong rem = gid % (inputH * inputW);
  ulong ih = rem / inputW;
  ulong iw = rem % inputW;

  TYPE sum = (TYPE)0;

  // Precompute filter base offset for this channel
  ulong filterCBase = filter_offset + c * filterH * filterW;

  if (strideY == 1 && strideX == 1) {
    // Optimized path for stride=1: no modulo needed
    for (ulong f = 0; f < numFilters; f++) {
      ulong gradFBase = grad_out_offset + f * outH * outW;
      ulong filterFBase = filterCBase + f * inputC * filterH * filterW;

      for (ulong kh = 0; kh < filterH; kh++) {
        long oh = (long)ih + (long)padY - (long)kh;

        if (oh < 0 || oh >= (long)outH)
          continue;

        ulong gradRowBase = gradFBase + (ulong)oh * outW;
        ulong filterRowBase = filterFBase + kh * filterW;

        for (ulong kw = 0; kw < filterW; kw++) {
          long ow = (long)iw + (long)padX - (long)kw;

          if (ow < 0 || ow >= (long)outW)
            continue;

          sum += grads[gradRowBase + (ulong)ow] * filters[filterRowBase + kw];
        }
      }
    }
  } else {
    // General path with stride support
    for (ulong f = 0; f < numFilters; f++) {
      ulong gradFBase = grad_out_offset + f * outH * outW;
      ulong filterFBase = filterCBase + f * inputC * filterH * filterW;

      for (ulong kh = 0; kh < filterH; kh++) {
        long numerator_h = (long)ih + (long)padY - (long)kh;

        if (numerator_h < 0 || numerator_h % (long)strideY != 0)
          continue;
        ulong oh = (ulong)numerator_h / strideY;

        if (oh >= outH)
          continue;

        ulong gradRowBase = gradFBase + oh * outW;
        ulong filterRowBase = filterFBase + kh * filterW;

        for (ulong kw = 0; kw < filterW; kw++) {
          long numerator_w = (long)iw + (long)padX - (long)kw;

          if (numerator_w < 0 || numerator_w % (long)strideX != 0)
            continue;
          ulong ow = (ulong)numerator_w / strideX;

          if (ow >= outW)
            continue;

          sum += grads[gradRowBase + ow] * filters[filterRowBase + kw];
        }
      }
    }
  }

  grads[grad_in_offset + gid] = sum;
}

//===================================================================================================================//

#endif // CNN_BACKPROPAGATE_CPP_CL
