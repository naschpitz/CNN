#ifndef CNN_KERNELS_CPP_CL
#define CNN_KERNELS_CPP_CL

// Note: Depends on CNN_Defines.hpp.cl (loaded first by C++ code)

//===================================================================================================================//
// Convolution kernels
//===================================================================================================================//

// Computes z = input ⊛ filter + bias (forward pass)
// Work items = batchSize * numFilters * outH * outW
kernel void calculate_conv2d(
    global TYPE* actvs,
    global TYPE* filters,
    global TYPE* biases,
    ulong actv_in_offset,    // offset of input tensor (per-sample)
    ulong actv_out_offset,   // offset of output tensor (per-sample)
    ulong filter_offset,     // offset of this layer's filters (shared)
    ulong bias_offset,       // offset of this layer's biases (shared)
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
    ulong batchSize,
    ulong actvStride         // totalActvSize — stride between samples in actvs
  ) {
  size_t gid = get_global_id(0);

  ulong totalOut = numFilters * outH * outW;
  ulong batchIdx = gid / totalOut;
  ulong localIdx = gid % totalOut;
  ulong batchOffset = batchIdx * actvStride;

  // Decompose localIdx into (f, oh, ow)
  ulong f  = localIdx / (outH * outW);
  ulong rem = localIdx % (outH * outW);
  ulong oh = rem / outW;
  ulong ow = rem % outW;

  TYPE sum = biases[bias_offset + f];

  for (ulong c = 0; c < inputC; c++) {
    for (ulong kh = 0; kh < filterH; kh++) {
      for (ulong kw = 0; kw < filterW; kw++) {
        long ih = (long)(oh * strideY + kh) - (long)padY;
        long iw = (long)(ow * strideX + kw) - (long)padX;

        if (ih >= 0 && ih < (long)inputH && iw >= 0 && iw < (long)inputW) {
          ulong input_idx = batchOffset + actv_in_offset + c * inputH * inputW + (ulong)ih * inputW + (ulong)iw;
          ulong filter_idx = filter_offset + f * inputC * filterH * filterW + c * filterH * filterW + kh * filterW + kw;
          sum += actvs[input_idx] * filters[filter_idx];
        }
      }
    }
  }

  actvs[batchOffset + actv_out_offset + f * outH * outW + oh * outW + ow] = sum;
}

//===================================================================================================================//

// Computes a = max(0, z) (forward pass)
// Work items = batchSize * size
kernel void calculate_relu(
    global TYPE* actvs,
    ulong in_offset,
    ulong out_offset,
    ulong size,
    ulong batchSize,
    ulong actvStride
  ) {
  size_t gid = get_global_id(0);
  ulong batchIdx = gid / size;
  ulong localIdx = gid % size;
  ulong batchOffset = batchIdx * actvStride;

  TYPE val = actvs[batchOffset + in_offset + localIdx];
  actvs[batchOffset + out_offset + localIdx] = (val > (TYPE)0) ? val : (TYPE)0;
}

//===================================================================================================================//

// Computes a = max(region) and records max indices (forward pass)
// Work items = batchSize * C * outH * outW
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
    ulong outW,
    ulong batchSize,
    ulong actvStride,
    ulong poolIdxStride
  ) {
  size_t gid = get_global_id(0);

  ulong totalOut = channels * outH * outW;
  ulong batchIdx = gid / totalOut;
  ulong localIdx = gid % totalOut;
  ulong batchActvOffset = batchIdx * actvStride;
  ulong batchPoolOffset = batchIdx * poolIdxStride;

  ulong c  = localIdx / (outH * outW);
  ulong rem = localIdx % (outH * outW);
  ulong oh = rem / outW;
  ulong ow = rem % outW;

  TYPE maxVal = -1e30f;
  ulong maxIdx = 0;

  for (ulong ph = 0; ph < poolH; ph++) {
    for (ulong pw = 0; pw < poolW; pw++) {
      ulong ih = oh * strideY + ph;
      ulong iw = ow * strideX + pw;

      // Absolute index into batched actvs buffer (used by backward pass too)
      ulong idx = batchActvOffset + actv_in_offset + c * inputH * inputW + ih * inputW + iw;
      TYPE val = actvs[idx];

      if (val > maxVal) {
        maxVal = val;
        maxIdx = idx;
      }
    }
  }

  actvs[batchActvOffset + actv_out_offset + localIdx] = maxVal;
  pool_indices[batchPoolOffset + pool_idx_offset + localIdx] = maxIdx;
}

//===================================================================================================================//
// Utility kernels
//===================================================================================================================//

// Zero a region of a buffer
// Work items = batchSize * size
kernel void zero_buffer(
    global TYPE* buf,
    ulong offset,
    ulong size,
    ulong batchSize,
    ulong stride
  ) {
  size_t gid = get_global_id(0);
  ulong batchIdx = gid / size;
  ulong localIdx = gid % size;

  buf[batchIdx * stride + offset + localIdx] = (TYPE)0;
}

//===================================================================================================================//
// Gradient kernels
//===================================================================================================================//

// Computes ∂Cost/∂maxpool_input: routes gradient to max position (backward pass)
// Work items = batchSize * size
kernel void calculate_dCost_dMaxpool(
    global TYPE* grads,
    global ulong* pool_indices,
    ulong grad_out_offset,     // offset of pool output gradient (per-sample)
    ulong pool_idx_offset,     // offset into pool_indices (per-sample)
    ulong size,                // C * outH * outW
    ulong batchSize,
    ulong gradStride,
    ulong poolIdxStride
  ) {
  size_t gid = get_global_id(0);
  ulong batchIdx = gid / size;
  ulong localIdx = gid % size;
  ulong batchGradOffset = batchIdx * gradStride;
  ulong batchPoolOffset = batchIdx * poolIdxStride;

  TYPE dOutVal = grads[batchGradOffset + grad_out_offset + localIdx];
  // maxIdx is an absolute index into the batched grads buffer (includes batch offset)
  ulong maxIdx = pool_indices[batchPoolOffset + pool_idx_offset + localIdx];

  grads[maxIdx] += dOutVal;
}

//===================================================================================================================//

// Computes ∂Cost/∂z = ∂Cost/∂a · (z > 0) (backward pass)
// Work items = batchSize * size
kernel void calculate_dCost_dRelu(
    global TYPE* grads,
    global TYPE* actvs,
    ulong grad_in_offset,    // offset to write input gradient
    ulong grad_out_offset,   // offset to read output gradient
    ulong actv_in_offset,    // offset of forward input activations
    ulong size,
    ulong batchSize,
    ulong actvStride          // same stride for actvs and grads
  ) {
  size_t gid = get_global_id(0);
  ulong batchIdx = gid / size;
  ulong localIdx = gid % size;
  ulong batchOffset = batchIdx * actvStride;

  TYPE actv = actvs[batchOffset + actv_in_offset + localIdx];
  TYPE dOut = grads[batchOffset + grad_out_offset + localIdx];

  grads[batchOffset + grad_in_offset + localIdx] = (actv > (TYPE)0) ? dOut : (TYPE)0;
}

//===================================================================================================================//

// Computes ∂Cost/∂W for convolution filters (backward pass)
// Work items = batchSize * numFilters * inputC * filterH * filterW
kernel void calculate_dCost_dFilters(
    global TYPE* grads,
    global TYPE* actvs,
    global TYPE* dFilters,
    ulong grad_out_offset,   // offset of output gradient (per-sample)
    ulong actv_in_offset,    // offset of forward input (per-sample)
    ulong dfilter_offset,    // offset in dFilters buffer (per-sample)
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
    ulong batchSize,
    ulong actvStride,
    ulong dFilterStride
  ) {
  size_t gid = get_global_id(0);

  ulong totalFilterElems = numFilters * inputC * filterH * filterW;
  ulong batchIdx = gid / totalFilterElems;
  ulong localIdx = gid % totalFilterElems;
  ulong batchActvOffset = batchIdx * actvStride;
  ulong batchDFilterOffset = batchIdx * dFilterStride;

  // Decompose localIdx into (f, c, kh, kw)
  ulong f   = localIdx / (inputC * filterH * filterW);
  ulong rem = localIdx % (inputC * filterH * filterW);
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
        TYPE dOut = grads[batchActvOffset + grad_out_offset + f * outH * outW + oh * outW + ow];
        TYPE inp = actvs[batchActvOffset + actv_in_offset + c * inputH * inputW + (ulong)ih * inputW + (ulong)iw];
        sum += dOut * inp;
      }
    }
  }

  dFilters[batchDFilterOffset + dfilter_offset + localIdx] = sum;
}

//===================================================================================================================//

// Computes ∂Cost/∂b for convolution biases (backward pass)
// Work items = batchSize * numFilters
kernel void calculate_dCost_dBiases(
    global TYPE* grads,
    global TYPE* dBiases,
    ulong grad_out_offset,
    ulong dbias_offset,
    ulong numFilters,
    ulong outH,
    ulong outW,
    ulong batchSize,
    ulong gradStride,
    ulong dBiasStride
  ) {
  size_t gid = get_global_id(0);
  ulong batchIdx = gid / numFilters;
  ulong f = gid % numFilters;
  ulong batchGradOffset = batchIdx * gradStride;
  ulong batchDBiasOffset = batchIdx * dBiasStride;

  TYPE sum = (TYPE)0;

  for (ulong oh = 0; oh < outH; oh++) {
    for (ulong ow = 0; ow < outW; ow++) {
      sum += grads[batchGradOffset + grad_out_offset + f * outH * outW + oh * outW + ow];
    }
  }

  dBiases[batchDBiasOffset + dbias_offset + f] = sum;
}

//===================================================================================================================//

// Computes ∂Cost/∂input via transposed convolution (backward pass)
// Work items = batchSize * inputC * inputH * inputW
kernel void calculate_dCost_dInput(
    global TYPE* grads,
    global TYPE* filters,
    ulong grad_out_offset,   // offset of conv output gradient (per-sample)
    ulong grad_in_offset,    // offset to write input gradient (per-sample)
    ulong filter_offset,     // offset of this layer's filters (shared)
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
    ulong batchSize,
    ulong gradStride
  ) {
  size_t gid = get_global_id(0);

  ulong totalInput = inputC * inputH * inputW;
  ulong batchIdx = gid / totalInput;
  ulong localIdx = gid % totalInput;
  ulong batchGradOffset = batchIdx * gradStride;

  // Decompose localIdx into (c, ih, iw)
  ulong c  = localIdx / (inputH * inputW);
  ulong rem = localIdx % (inputH * inputW);
  ulong ih = rem / inputW;
  ulong iw = rem % inputW;

  TYPE sum = (TYPE)0;

  for (ulong f = 0; f < numFilters; f++) {
    for (ulong kh = 0; kh < filterH; kh++) {
      for (ulong kw = 0; kw < filterW; kw++) {
        long numerator_h = (long)ih + (long)padY - (long)kh;
        long numerator_w = (long)iw + (long)padX - (long)kw;

        if (numerator_h >= 0 && numerator_w >= 0 &&
            numerator_h % (long)strideY == 0 && numerator_w % (long)strideX == 0) {
          ulong oh = (ulong)numerator_h / strideY;
          ulong ow = (ulong)numerator_w / strideX;

          if (oh < outH && ow < outW) {
            TYPE dOut = grads[batchGradOffset + grad_out_offset + f * outH * outW + oh * outW + ow];
            TYPE filt = filters[filter_offset + f * inputC * filterH * filterW + c * filterH * filterW + kh * filterW + kw];
            sum += dOut * filt;
          }
        }
      }
    }
  }

  grads[batchGradOffset + grad_in_offset + localIdx] = sum;
}

//===================================================================================================================//
// Accumulate and Update kernels
//===================================================================================================================//

// Accumulates per-sample gradients: reduces across batch dimension
// Work items = size (NOT batched — each work item sums over batchSize samples)
kernel void accumulate_gradients(
    global TYPE* accum,
    global TYPE* grad,
    ulong offset,
    ulong size,
    ulong batchSize,
    ulong gradStride
  ) {
  size_t gid = get_global_id(0);
  if (gid >= size) return;

  TYPE sum = (TYPE)0;
  for (ulong b = 0; b < batchSize; b++) {
    sum += grad[b * gradStride + offset + gid];
  }
  accum[offset + gid] += sum;
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
// Both are batch-aware: each sample's CNN output → corresponding ANN input slot.
//===================================================================================================================//

__kernel void copy_cnn_to_ann(
    __global const TYPE* cnn_actvs,
    __global TYPE* actvs,
    const ulong cnn_offset,
    const ulong size,
    const ulong batchSize,
    const ulong cnnStride,
    const ulong annStride) {
  ulong gid = get_global_id(0);

  ulong batchIdx = gid / size;
  ulong localIdx = gid % size;

  actvs[batchIdx * annStride + localIdx] = cnn_actvs[batchIdx * cnnStride + cnn_offset + localIdx];
}

//===================================================================================================================//

__kernel void copy_ann_grad_to_cnn(
    __global const TYPE* dCost_dActvs,
    __global TYPE* cnn_grads,
    const ulong cnn_offset,
    const ulong size,
    const ulong batchSize,
    const ulong cnnStride,
    const ulong annStride) {
  ulong gid = get_global_id(0);

  ulong batchIdx = gid / size;
  ulong localIdx = gid % size;

  cnn_grads[batchIdx * cnnStride + cnn_offset + localIdx] = dCost_dActvs[batchIdx * annStride + localIdx];
}

//===================================================================================================================//

#endif // CNN_KERNELS_CPP_CL

