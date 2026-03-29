#ifndef CNN_RESIDUAL_CPP_CL
#define CNN_RESIDUAL_CPP_CL

//===================================================================================================================//
// CNN_Residual.cpp.cl
//
// Residual (skip) connection kernels.
// Identity shortcut: out[i] += skip[i]
// Projection shortcut: out[i] += bias[oc] + sum_ic(W[oc*inC+ic] * skip[ic*spatial+s])
// Backward identity: dSkip[i] = dOut[i]
// Backward projection: dSkip = W^T * dOut, dW += dOut * skip^T, dB += sum(dOut)
//===================================================================================================================//

//===================================================================================================================//
//-- Forward pass --//
//===================================================================================================================//

// Identity shortcut: element-wise add. One work-item per element.
kernel void residual_add(global TYPE* actvs, ulong outOffset, ulong skipOffset, ulong size)
{
  size_t gid = get_global_id(0);

  if (gid >= size)
    return;

  actvs[outOffset + gid] += actvs[skipOffset + gid];
}

//===================================================================================================================//

// Projection shortcut: 1x1 conv. One work-item per output element (oc, s).
kernel void residual_add_proj(global TYPE* actvs, ulong outOffset, ulong skipOffset,
                              global const TYPE* projW, global const TYPE* projB,
                              ulong inC, ulong outC, ulong spatialSize)
{
  size_t gid = get_global_id(0);
  ulong totalOut = outC * spatialSize;

  if (gid >= totalOut)
    return;

  ulong oc = gid / spatialSize;
  ulong s = gid % spatialSize;

  TYPE projected = projB[oc];

  for (ulong ic = 0; ic < inC; ic++)
    projected += projW[oc * inC + ic] * actvs[skipOffset + ic * spatialSize + s];

  actvs[outOffset + gid] += projected;
}

//===================================================================================================================//
//-- Backward pass --//
//===================================================================================================================//

// Identity backward: copy dOut to dSkip. One work-item per element.
kernel void residual_bwd(global TYPE* grads, ulong dSkipOffset, ulong dOutOffset, ulong size)
{
  size_t gid = get_global_id(0);

  if (gid >= size)
    return;

  // dSkip = dOut (gradient passes through unchanged; the block gradient continues as-is)
  grads[dSkipOffset + gid] += grads[dOutOffset + gid];
}

//===================================================================================================================//

// Projection backward: compute dSkip = W^T * dOut. One work-item per skip element (ic, s).
kernel void residual_bwd_proj_dskip(global TYPE* grads, global const TYPE* projW,
                                     ulong dSkipOffset, ulong dOutOffset,
                                     ulong inC, ulong outC, ulong spatialSize)
{
  size_t gid = get_global_id(0);
  ulong totalIn = inC * spatialSize;

  if (gid >= totalIn)
    return;

  ulong ic = gid / spatialSize;
  ulong s = gid % spatialSize;

  TYPE sum = (TYPE)0;

  for (ulong oc = 0; oc < outC; oc++)
    sum += projW[oc * inC + ic] * grads[dOutOffset + oc * spatialSize + s];

  grads[dSkipOffset + gid] += sum;
}

//===================================================================================================================//

// Projection backward: accumulate weight gradients. One work-item per weight (oc, ic).
kernel void residual_bwd_proj_dw(global const TYPE* grads, global const TYPE* actvs,
                                  global TYPE* dProjW, global TYPE* dProjB,
                                  ulong dOutOffset, ulong skipOffset,
                                  ulong inC, ulong outC, ulong spatialSize)
{
  size_t gid = get_global_id(0);

  if (gid >= outC * inC)
    return;

  ulong oc = gid / inC;
  ulong ic = gid % inC;

  TYPE dw = (TYPE)0;

  for (ulong s = 0; s < spatialSize; s++)
    dw += grads[dOutOffset + oc * spatialSize + s] * actvs[skipOffset + ic * spatialSize + s];

  dProjW[gid] += dw;

  // Accumulate bias gradient (only for ic == 0 to avoid duplicate adds)
  if (ic == 0) {
    TYPE db = (TYPE)0;

    for (ulong s = 0; s < spatialSize; s++)
      db += grads[dOutOffset + oc * spatialSize + s];

    dProjB[oc] += db;
  }
}

//===================================================================================================================//

#endif // CNN_RESIDUAL_CPP_CL

