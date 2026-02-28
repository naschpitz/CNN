#ifndef CNN_KERNELS_CPP_CL
#define CNN_KERNELS_CPP_CL

// Note: Depends on CNN_Defines.hpp.cl (loaded first by C++ code)

//===================================================================================================================//
// Convolution kernels
//===================================================================================================================//

// Computes z = input ⊛ filter + bias (forward pass)
// One work-item per output element (f, oh, ow)
// nElements = numFilters * outH * outW
kernel void calculate_conv2d(
    global TYPE* actvs,
    global TYPE* filters,
    global TYPE* biases,
    ulong actv_in_offset,    // offset of input tensor in actvs buffer
    ulong actv_out_offset,   // offset of output tensor in actvs buffer
    ulong filter_offset,     // offset of this layer's filters
    ulong bias_offset,       // offset of this layer's biases
    ulong inputC,
    ulong inputH,
    ulong inputW,
    ulong numFilters,
    ulong filterH,
    ulong filterW,
    ulong strideY,
    ulong strideX,
    ulong padY,
    ulong padX,
    ulong outH,
    ulong outW
  ) {
  size_t gid = get_global_id(0);

  ulong totalOut = numFilters * outH * outW;
  if (gid >= totalOut) return;

  // Decompose gid into (f, oh, ow)
  ulong f  = gid / (outH * outW);
  ulong rem = gid % (outH * outW);
  ulong oh = rem / outW;
  ulong ow = rem % outW;

  TYPE sum = biases[bias_offset + f];

  for (ulong c = 0; c < inputC; c++) {
    for (ulong kh = 0; kh < filterH; kh++) {
      for (ulong kw = 0; kw < filterW; kw++) {
        long ih = (long)(oh * strideY + kh) - (long)padY;
        long iw = (long)(ow * strideX + kw) - (long)padX;

        if (ih >= 0 && ih < (long)inputH && iw >= 0 && iw < (long)inputW) {
          ulong input_idx = actv_in_offset + c * inputH * inputW + (ulong)ih * inputW + (ulong)iw;
          ulong filter_idx = filter_offset + f * inputC * filterH * filterW + c * filterH * filterW + kh * filterW + kw;
          sum += actvs[input_idx] * filters[filter_idx];
        }
      }
    }
  }

  actvs[actv_out_offset + f * outH * outW + oh * outW + ow] = sum;
}

//===================================================================================================================//

// Computes a = max(0, z) (forward pass)
// One work-item per element, nElements = layer tensor size
kernel void calculate_relu(
    global TYPE* actvs,
    ulong in_offset,
    ulong out_offset,
    ulong size
  ) {
  size_t gid = get_global_id(0);
  if (gid >= size) return;

  TYPE val = actvs[in_offset + gid];
  actvs[out_offset + gid] = (val > (TYPE)0) ? val : (TYPE)0;
}

//===================================================================================================================//

// Computes a = max(region) and records max indices (forward pass)
// One work-item per output element (c, oh, ow)
// nElements = C * outH * outW
kernel void calculate_maxpool(
    global TYPE* actvs,
    global ulong* pool_indices,
    ulong actv_in_offset,
    ulong actv_out_offset,
    ulong pool_idx_offset,
    ulong channels,
    ulong inputH,
    ulong inputW,
    ulong poolH,
    ulong poolW,
    ulong strideY,
    ulong strideX,
    ulong outH,
    ulong outW
  ) {
  size_t gid = get_global_id(0);

  ulong totalOut = channels * outH * outW;
  if (gid >= totalOut) return;

  ulong c  = gid / (outH * outW);
  ulong rem = gid % (outH * outW);
  ulong oh = rem / outW;
  ulong ow = rem % outW;

  TYPE maxVal = -1e30f;
  ulong maxIdx = 0;

  for (ulong ph = 0; ph < poolH; ph++) {
    for (ulong pw = 0; pw < poolW; pw++) {
      ulong ih = oh * strideY + ph;
      ulong iw = ow * strideX + pw;

      ulong idx = actv_in_offset + c * inputH * inputW + ih * inputW + iw;
      TYPE val = actvs[idx];

      if (val > maxVal) {
        maxVal = val;
        maxIdx = idx;
      }
    }
  }

  actvs[actv_out_offset + gid] = maxVal;
  pool_indices[pool_idx_offset + gid] = maxIdx;
}

//===================================================================================================================//
// Utility kernels
//===================================================================================================================//

// Zero a region of a buffer
kernel void zero_buffer(
    global TYPE* buf,
    ulong offset,
    ulong size
  ) {
  size_t gid = get_global_id(0);
  if (gid >= size) return;

  buf[offset + gid] = (TYPE)0;
}

//===================================================================================================================//
// Gradient kernels
//===================================================================================================================//

// Computes ∂Cost/∂maxpool_input: routes gradient to max position (backward pass)
// nElements = C * outH * outW (same as forward output size)
kernel void calculate_dCost_dMaxpool(
    global TYPE* grads,
    global ulong* pool_indices,
    ulong grad_out_offset,     // offset of pool output gradient (source)
    ulong pool_idx_offset,     // offset into pool_indices for this layer
    ulong size                 // C * outH * outW
  ) {
  size_t gid = get_global_id(0);
  if (gid >= size) return;

  TYPE dOutVal = grads[grad_out_offset + gid];
  ulong maxIdx = pool_indices[pool_idx_offset + gid];

  // Route gradient to the max position in the input gradient buffer
  // maxIdx is an absolute index into the actvs/grads buffer
  grads[maxIdx] += dOutVal;
}

//===================================================================================================================//

// Computes ∂Cost/∂z = ∂Cost/∂a · (z > 0) (backward pass)
// One work-item per element
kernel void calculate_dCost_dRelu(
    global TYPE* grads,
    global TYPE* actvs,
    ulong grad_in_offset,    // offset to write input gradient
    ulong grad_out_offset,   // offset to read output gradient
    ulong actv_in_offset,    // offset of forward input activations
    ulong size
  ) {
  size_t gid = get_global_id(0);
  if (gid >= size) return;

  TYPE actv = actvs[actv_in_offset + gid];
  TYPE dOut = grads[grad_out_offset + gid];

  grads[grad_in_offset + gid] = (actv > (TYPE)0) ? dOut : (TYPE)0;
}

//===================================================================================================================//

// Computes ∂Cost/∂W for convolution filters (backward pass)
// One work-item per filter weight element: (f, c, kh, kw)
// nElements = numFilters * inputC * filterH * filterW
kernel void calculate_dCost_dFilters(
    global TYPE* grads,
    global TYPE* actvs,
    global TYPE* dFilters,
    ulong grad_out_offset,   // offset of output gradient in grads buffer
    ulong actv_in_offset,    // offset of forward input in actvs buffer
    ulong dfilter_offset,    // offset in dFilters buffer for this layer
    ulong inputC,
    ulong inputH,
    ulong inputW,
    ulong numFilters,
    ulong filterH,
    ulong filterW,
    ulong strideY,
    ulong strideX,
    ulong padY,
    ulong padX,
    ulong outH,
    ulong outW
  ) {
  size_t gid = get_global_id(0);

  ulong totalFilterElems = numFilters * inputC * filterH * filterW;
  if (gid >= totalFilterElems) return;

  // Decompose gid into (f, c, kh, kw)
  ulong f   = gid / (inputC * filterH * filterW);
  ulong rem = gid % (inputC * filterH * filterW);
  ulong c   = rem / (filterH * filterW);
  ulong rem2 = rem % (filterH * filterW);
  ulong kh  = rem2 / filterW;
  ulong kw  = rem2 % filterW;

  TYPE sum = (TYPE)0;

  for (ulong oh = 0; oh < outH; oh++) {
    for (ulong ow = 0; ow < outW; ow++) {
      long ih = (long)(oh * strideY + kh) - (long)padY;
      long iw = (long)(ow * strideX + kw) - (long)padX;

      if (ih >= 0 && ih < (long)inputH && iw >= 0 && iw < (long)inputW) {
        TYPE dOut = grads[grad_out_offset + f * outH * outW + oh * outW + ow];
        TYPE inp = actvs[actv_in_offset + c * inputH * inputW + (ulong)ih * inputW + (ulong)iw];
        sum += dOut * inp;
      }
    }
  }

  dFilters[dfilter_offset + gid] = sum;
}

//===================================================================================================================//

// Tiled version of calculate_dCost_dFilters for layers with few filter elements.
// Parallelized over (f, c, kh, kw, tile) where tile partitions the output spatial dims.
// nElements = numFilters * inputC * filterH * filterW * numTiles
// Requires dFilters to be zeroed before invocation.
kernel void calculate_dCost_dFilters_tiled(
    global TYPE* grads,
    global TYPE* actvs,
    global TYPE* dFilters,
    ulong grad_out_offset,
    ulong actv_in_offset,
    ulong dfilter_offset,
    ulong inputC,
    ulong inputH,
    ulong inputW,
    ulong numFilters,
    ulong filterH,
    ulong filterW,
    ulong strideY,
    ulong strideX,
    ulong padY,
    ulong padX,
    ulong outH,
    ulong outW,
    ulong tileSize,
    ulong numTiles
  ) {
  size_t gid = get_global_id(0);

  ulong totalFilterElems = numFilters * inputC * filterH * filterW;
  ulong totalWork = totalFilterElems * numTiles;
  if (gid >= totalWork) return;

  ulong filterElemIdx = gid / numTiles;
  ulong tileIdx = gid % numTiles;

  ulong f   = filterElemIdx / (inputC * filterH * filterW);
  ulong rem = filterElemIdx % (inputC * filterH * filterW);
  ulong c   = rem / (filterH * filterW);
  ulong rem2 = rem % (filterH * filterW);
  ulong kh  = rem2 / filterW;
  ulong kw  = rem2 % filterW;

  ulong totalOutPositions = outH * outW;
  ulong tileStart = tileIdx * tileSize;
  ulong tileEnd = tileStart + tileSize;
  if (tileEnd > totalOutPositions) tileEnd = totalOutPositions;

  TYPE sum = (TYPE)0;

  for (ulong pos = tileStart; pos < tileEnd; pos++) {
    ulong oh = pos / outW;
    ulong ow = pos % outW;

    long ih = (long)(oh * strideY + kh) - (long)padY;
    long iw = (long)(ow * strideX + kw) - (long)padX;

    if (ih >= 0 && ih < (long)inputH && iw >= 0 && iw < (long)inputW) {
      TYPE dOut = grads[grad_out_offset + f * outH * outW + oh * outW + ow];
      TYPE inp = actvs[actv_in_offset + c * inputH * inputW + (ulong)ih * inputW + (ulong)iw];
      sum += dOut * inp;
    }
  }

  // Atomic add since multiple tiles write to the same filter element
  global volatile TYPE* addr = (global volatile TYPE*)&dFilters[dfilter_offset + filterElemIdx];
  union { TYPE f; uint i; } expected, desired;
  do {
    expected.f = *addr;
    desired.f = expected.f + sum;
  } while (atomic_cmpxchg((global volatile uint*)addr, expected.i, desired.i) != expected.i);
}

//===================================================================================================================//

// Computes ∂Cost/∂b for convolution biases (backward pass)
// One work-item per filter: nElements = numFilters
kernel void calculate_dCost_dBiases(
    global TYPE* grads,
    global TYPE* dBiases,
    ulong grad_out_offset,
    ulong dbias_offset,
    ulong numFilters,
    ulong outH,
    ulong outW
  ) {
  size_t gid = get_global_id(0);
  if (gid >= numFilters) return;

  ulong f = gid;
  ulong totalOutPositions = outH * outW;
  TYPE sum = (TYPE)0;

  for (ulong pos = 0; pos < totalOutPositions; pos++) {
    sum += grads[grad_out_offset + f * totalOutPositions + pos];
  }

  dBiases[dbias_offset + f] = sum;
}

//===================================================================================================================//

// Tiled version of calculate_dCost_dBiases for layers with few filters.
// Parallelized over (f, tile). Requires dBiases to be zeroed before invocation.
// nElements = numFilters * numTiles
kernel void calculate_dCost_dBiases_tiled(
    global TYPE* grads,
    global TYPE* dBiases,
    ulong grad_out_offset,
    ulong dbias_offset,
    ulong numFilters,
    ulong outH,
    ulong outW,
    ulong tileSize,
    ulong numTiles
  ) {
  size_t gid = get_global_id(0);

  ulong totalWork = numFilters * numTiles;
  if (gid >= totalWork) return;

  ulong f = gid / numTiles;
  ulong tileIdx = gid % numTiles;

  ulong totalOutPositions = outH * outW;
  ulong tileStart = tileIdx * tileSize;
  ulong tileEnd = tileStart + tileSize;
  if (tileEnd > totalOutPositions) tileEnd = totalOutPositions;

  TYPE sum = (TYPE)0;

  for (ulong pos = tileStart; pos < tileEnd; pos++) {
    sum += grads[grad_out_offset + f * totalOutPositions + pos];
  }

  // Atomic add since multiple tiles write to the same bias
  global volatile TYPE* addr = (global volatile TYPE*)&dBiases[dbias_offset + f];
  union { TYPE f; uint i; } expected, desired;
  do {
    expected.f = *addr;
    desired.f = expected.f + sum;
  } while (atomic_cmpxchg((global volatile uint*)addr, expected.i, desired.i) != expected.i);
}

//===================================================================================================================//

// Computes ∂Cost/∂input via transposed convolution (backward pass)
// One work-item per input element: (c, ih, iw)
// nElements = inputC * inputH * inputW
kernel void calculate_dCost_dInput(
    global TYPE* grads,
    global TYPE* filters,
    ulong grad_out_offset,   // offset of conv output gradient
    ulong grad_in_offset,    // offset to write input gradient
    ulong filter_offset,     // offset of this layer's filters
    ulong inputC,
    ulong inputH,
    ulong inputW,
    ulong numFilters,
    ulong filterH,
    ulong filterW,
    ulong strideY,
    ulong strideX,
    ulong padY,
    ulong padX,
    ulong outH,
    ulong outW
  ) {
  size_t gid = get_global_id(0);

  ulong totalInput = inputC * inputH * inputW;
  if (gid >= totalInput) return;

  // Decompose gid into (c, ih, iw)
  ulong c  = gid / (inputH * inputW);
  ulong rem = gid % (inputH * inputW);
  ulong ih = rem / inputW;
  ulong iw = rem % inputW;

  TYPE sum = (TYPE)0;

  for (ulong f = 0; f < numFilters; f++) {
    for (ulong kh = 0; kh < filterH; kh++) {
      for (ulong kw = 0; kw < filterW; kw++) {
        // Reverse the forward mapping: oh * strideY + kh - padY = ih
        // So oh = (ih + padY - kh) / strideY, valid when (ih + padY - kh) % strideY == 0
        long numerator_h = (long)ih + (long)padY - (long)kh;
        long numerator_w = (long)iw + (long)padX - (long)kw;

        if (numerator_h >= 0 && numerator_w >= 0 &&
            numerator_h % (long)strideY == 0 && numerator_w % (long)strideX == 0) {
          ulong oh = (ulong)numerator_h / strideY;
          ulong ow = (ulong)numerator_w / strideX;

          if (oh < outH && ow < outW) {
            TYPE dOut = grads[grad_out_offset + f * outH * outW + oh * outW + ow];
            TYPE filt = filters[filter_offset + f * inputC * filterH * filterW + c * filterH * filterW + kh * filterW + kw];
            sum += dOut * filt;
          }
        }
      }
    }
  }

  grads[grad_in_offset + gid] = sum;
}

//===================================================================================================================//
// Accumulate and Update kernels
//===================================================================================================================//

// Accumulates per-sample gradients: accum[gid] += grad[gid]
kernel void accumulate_gradients(
    global TYPE* accum,
    global TYPE* grad,
    ulong offset,
    ulong size
  ) {
  size_t gid = get_global_id(0);
  if (gid >= size) return;

  accum[offset + gid] += grad[offset + gid];
}

//===================================================================================================================//

// Updates parameters: params[gid] -= lr · (accum[gid] / numSamples)
kernel void update_parameters(
    global TYPE* params,
    global TYPE* accum,
    ulong offset,
    ulong size,
    ulong numSamples,
    float learningRate
  ) {
  size_t gid = get_global_id(0);
  if (gid >= size) return;

  params[offset + gid] -= learningRate * (accum[offset + gid] / (TYPE)numSamples);
}

//===================================================================================================================//
// Bridge kernels: copy data between CNN and ANN buffers on the same OpenCL core.
// copy_cnn_to_ann: copies the last CNN activation (flatten output) into ANN's actvs buffer (layer 0).
// copy_ann_grad_to_cnn: copies ANN's input gradients (dCost_dActvs, layer 0) into CNN's gradient buffer.
//===================================================================================================================//

__kernel void copy_cnn_to_ann(
    __global const TYPE* cnn_actvs,
    __global TYPE* actvs,
    const ulong cnn_offset,
    const ulong size) {
  ulong gid = get_global_id(0);

  if (gid >= size) return;

  actvs[gid] = cnn_actvs[cnn_offset + gid];
}

//===================================================================================================================//

__kernel void copy_ann_grad_to_cnn(
    __global const TYPE* dCost_dActvs,
    __global TYPE* cnn_grads,
    const ulong cnn_offset,
    const ulong size) {
  ulong gid = get_global_id(0);

  if (gid >= size) return;

  cnn_grads[cnn_offset + gid] = dCost_dActvs[gid];
}

//===================================================================================================================//

#endif // CNN_KERNELS_CPP_CL

