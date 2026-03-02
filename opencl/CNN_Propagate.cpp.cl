#ifndef CNN_PROPAGATE_CPP_CL
#define CNN_PROPAGATE_CPP_CL

// Note: Depends on CNN_Defines.hpp.cl (TYPE)

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

  ulong filterFBase = filter_offset + f * inputC * filterH * filterW;

  for (ulong c = 0; c < inputC; c++) {
    ulong actvCBase = actv_in_offset + c * inputH * inputW;
    ulong filterCBase = filterFBase + c * filterH * filterW;

    for (ulong kh = 0; kh < filterH; kh++) {
      long ih = (long)(oh * strideY + kh) - (long)padY;
      if (ih < 0 || ih >= (long)inputH) continue;

      ulong actvRowBase = actvCBase + (ulong)ih * inputW;
      ulong filterRowBase = filterCBase + kh * filterW;

      for (ulong kw = 0; kw < filterW; kw++) {
        long iw = (long)(ow * strideX + kw) - (long)padX;
        if (iw < 0 || iw >= (long)inputW) continue;

        sum += actvs[actvRowBase + (ulong)iw] * filters[filterRowBase + kw];
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

#endif // CNN_PROPAGATE_CPP_CL

