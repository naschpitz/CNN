#ifndef CNN_PROPAGATE_CPP_CL
#define CNN_PROPAGATE_CPP_CL

// Note: Depends on CNN_Defines.hpp.cl (TYPE)

//===================================================================================================================//

// Computes a = max(0, z) (forward pass)
// One work-item per element, nElements = layer tensor size
kernel void calculate_relu(global TYPE* actvs, ulong in_offset, ulong out_offset, ulong size)
{
  size_t gid = get_global_id(0);

  if (gid >= size)
    return;

  TYPE val = actvs[in_offset + gid];
  actvs[out_offset + gid] = (val > (TYPE)0) ? val : (TYPE)0;
}

//===================================================================================================================//

// Computes a = max(region) and records max indices (forward pass)
// One work-item per output element (c, oh, ow)
// nElements = C * outH * outW
kernel void calculate_maxpool(global TYPE* actvs, global ulong* pool_indices, ulong actv_in_offset,
                              ulong actv_out_offset, ulong pool_idx_offset, ulong channels, ulong inputH, ulong inputW,
                              ulong poolH, ulong poolW, ulong strideY, ulong strideX, ulong outH, ulong outW)
{
  size_t gid = get_global_id(0);

  ulong totalOut = channels * outH * outW;

  if (gid >= totalOut)
    return;

  ulong c = gid / (outH * outW);
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

// Computes a = avg(region) (forward pass)
// One work-item per output element (c, oh, ow)
// nElements = C * outH * outW
kernel void calculate_avgpool(global TYPE* actvs, ulong actv_in_offset, ulong actv_out_offset, ulong channels,
                              ulong inputH, ulong inputW, ulong poolH, ulong poolW, ulong strideY, ulong strideX,
                              ulong outH, ulong outW)
{
  size_t gid = get_global_id(0);

  ulong totalOut = channels * outH * outW;

  if (gid >= totalOut)
    return;

  ulong c = gid / (outH * outW);
  ulong rem = gid % (outH * outW);
  ulong oh = rem / outW;
  ulong ow = rem % outW;

  TYPE sum = (TYPE)0;

  for (ulong ph = 0; ph < poolH; ph++) {
    for (ulong pw = 0; pw < poolW; pw++) {
      ulong ih = oh * strideY + ph;
      ulong iw = ow * strideX + pw;

      sum += actvs[actv_in_offset + c * inputH * inputW + ih * inputW + iw];
    }
  }

  actvs[actv_out_offset + gid] = sum / (TYPE)(poolH * poolW);
}

//===================================================================================================================//

#endif // CNN_PROPAGATE_CPP_CL
