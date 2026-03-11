#ifndef CNN_BATCHNORM_CPP_CL
#define CNN_BATCHNORM_CPP_CL

// Note: Depends on CNN_Defines.hpp.cl (TYPE)
// BatchNorm GPU kernels — currently reuse the same per-sample spatial normalization
// as InstanceNorm. True cross-sample batch normalization on GPU requires a
// layer-by-layer multi-sample pipeline (future work). The kernel names are
// prefixed with "batchnorm_" to keep them separate from InstanceNorm kernels.

//===================================================================================================================//

// BatchNorm forward: compute per-channel mean over spatial dims (single sample)
// One work-group per channel. Uses tree reduction.
kernel void batchnorm_compute_mean(global TYPE* actvs, global TYPE* bn_batch_mean, ulong actv_in_offset,
                                   ulong norm_param_offset, ulong C, ulong H, ulong W)
{
  local TYPE partials[256];

  size_t groupId = get_group_id(0);
  size_t lid = get_local_id(0);
  size_t localSize = get_local_size(0);

  ulong c = groupId;
  ulong spatialSize = H * W;
  ulong base = actv_in_offset + c * spatialSize;

  TYPE sum = (TYPE)0;

  for (ulong s = lid; s < spatialSize; s += localSize) {
    sum += actvs[base + s];
  }

  partials[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (ulong stride = localSize / 2; stride > 0; stride >>= 1) {
    if (lid < stride)
      partials[lid] += partials[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    bn_batch_mean[norm_param_offset + c] = partials[0] / (TYPE)spatialSize;
  }
}

//===================================================================================================================//

// BatchNorm forward: compute per-channel variance over spatial dims (single sample)
kernel void batchnorm_compute_var(global TYPE* actvs, global TYPE* bn_batch_mean, global TYPE* bn_batch_var,
                                  ulong actv_in_offset, ulong norm_param_offset, ulong C, ulong H, ulong W)
{
  local TYPE partials[256];

  size_t groupId = get_group_id(0);
  size_t lid = get_local_id(0);
  size_t localSize = get_local_size(0);

  ulong c = groupId;
  ulong spatialSize = H * W;
  ulong base = actv_in_offset + c * spatialSize;
  TYPE mean = bn_batch_mean[norm_param_offset + c];

  TYPE sum = (TYPE)0;

  for (ulong s = lid; s < spatialSize; s += localSize) {
    TYPE diff = actvs[base + s] - mean;
    sum += diff * diff;
  }

  partials[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (ulong stride = localSize / 2; stride > 0; stride >>= 1) {
    if (lid < stride)
      partials[lid] += partials[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    bn_batch_var[norm_param_offset + c] = partials[0] / (TYPE)spatialSize;
  }
}

//===================================================================================================================//

// BatchNorm forward: normalize, scale, shift, and store xnorm
kernel void batchnorm_normalize(global TYPE* actvs, global TYPE* bn_xnorm, global TYPE* bn_gamma, global TYPE* bn_beta,
                                global TYPE* bn_mean, global TYPE* bn_var, ulong actv_in_offset, ulong actv_out_offset,
                                ulong xnorm_offset, ulong norm_param_offset, ulong C, ulong H, ulong W, float epsilon)
{
  size_t gid = get_global_id(0);
  ulong spatialSize = H * W;
  ulong totalSize = C * spatialSize;

  if (gid >= totalSize)
    return;

  ulong c = gid / spatialSize;
  TYPE x = actvs[actv_in_offset + gid];
  TYPE mean = bn_mean[norm_param_offset + c];
  TYPE var = bn_var[norm_param_offset + c];
  TYPE gamma = bn_gamma[norm_param_offset + c];
  TYPE beta = bn_beta[norm_param_offset + c];

  TYPE invStd = (TYPE)1 / sqrt(var + (TYPE)epsilon);
  TYPE xn = (x - mean) * invStd;
  bn_xnorm[xnorm_offset + gid] = xn;
  actvs[actv_out_offset + gid] = gamma * xn + beta;
}

//===================================================================================================================//

// BatchNorm backward: compute dGamma and dBeta per channel (reduction)
kernel void batchnorm_dGammaBeta(global TYPE* grads, global TYPE* bn_xnorm, global TYPE* bn_dGamma,
                                 global TYPE* bn_dBeta, ulong grad_out_offset, ulong xnorm_offset,
                                 ulong norm_param_offset, ulong C, ulong H, ulong W)
{
  local TYPE partials_dg[256];
  local TYPE partials_db[256];

  size_t groupId = get_group_id(0);
  size_t lid = get_local_id(0);
  size_t localSize = get_local_size(0);

  ulong c = groupId;
  ulong spatialSize = H * W;

  TYPE sum_dg = (TYPE)0;
  TYPE sum_db = (TYPE)0;

  for (ulong s = lid; s < spatialSize; s += localSize) {
    ulong idx = c * spatialSize + s;
    TYPE dOut = grads[grad_out_offset + idx];
    TYPE xn = bn_xnorm[xnorm_offset + idx];
    sum_dg += dOut * xn;
    sum_db += dOut;
  }

  partials_dg[lid] = sum_dg;
  partials_db[lid] = sum_db;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (ulong stride = localSize / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      partials_dg[lid] += partials_dg[lid + stride];
      partials_db[lid] += partials_db[lid + stride];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    bn_dGamma[norm_param_offset + c] = partials_dg[0];
    bn_dBeta[norm_param_offset + c] = partials_db[0];
  }
}

//===================================================================================================================//

// BatchNorm backward: compute dInput
kernel void batchnorm_dInput(global TYPE* grads, global TYPE* bn_xnorm, global TYPE* bn_gamma, global TYPE* bn_dGamma,
                             global TYPE* bn_dBeta, global TYPE* bn_batch_var, ulong grad_in_offset,
                             ulong grad_out_offset, ulong xnorm_offset, ulong norm_param_offset, ulong C, ulong H,
                             ulong W, float epsilon)
{
  size_t gid = get_global_id(0);
  ulong spatialSize = H * W;
  ulong totalSize = C * spatialSize;

  if (gid >= totalSize)
    return;

  ulong c = gid / spatialSize;
  TYPE gamma = bn_gamma[norm_param_offset + c];
  TYPE var = bn_batch_var[norm_param_offset + c];
  TYPE invStd = (TYPE)1 / sqrt(var + (TYPE)epsilon);
  TYPE N = (TYPE)spatialSize;
  TYPE dg = bn_dGamma[norm_param_offset + c];
  TYPE db = bn_dBeta[norm_param_offset + c];
  TYPE xn = bn_xnorm[xnorm_offset + gid];
  TYPE dOut = grads[grad_out_offset + gid];

  grads[grad_in_offset + gid] = (gamma * invStd / N) * (N * dOut - db - xn * dg);
}

//===================================================================================================================//

// Update running stats for BatchNorm
kernel void batchnorm_update_running_stats(global TYPE* bn_running_mean, global TYPE* bn_running_var,
                                           global TYPE* accum_bn_batch_mean, global TYPE* accum_bn_batch_var,
                                           ulong norm_param_offset, ulong numChannels, float momentum, ulong numSamples)
{
  size_t gid = get_global_id(0);

  if (gid >= numChannels)
    return;

  ulong idx = norm_param_offset + gid;
  TYPE avgMean = accum_bn_batch_mean[idx] / (TYPE)numSamples;
  TYPE avgVar = accum_bn_batch_var[idx] / (TYPE)numSamples;
  bn_running_mean[idx] = ((TYPE)1 - (TYPE)momentum) * bn_running_mean[idx] + (TYPE)momentum * avgMean;
  bn_running_var[idx] = ((TYPE)1 - (TYPE)momentum) * bn_running_var[idx] + (TYPE)momentum * avgVar;
}

//===================================================================================================================//
//-- True batch-wide kernels (cross-sample reduction) --//
//===================================================================================================================//

// Compute per-channel mean across ALL N samples: mean_c = sum_{n,h,w} x[n,c,h,w] / (N*H*W)
// One work-group per channel. Reads from batch_actvs where sample n starts at n*sampleStride.
kernel void batchnorm_batch_compute_mean(global TYPE* batch_actvs, global TYPE* bn_batch_mean, ulong actv_in_offset,
                                         ulong norm_param_offset, ulong C, ulong H, ulong W, ulong N,
                                         ulong sampleStride)
{
  local TYPE partials[256];

  size_t groupId = get_group_id(0);
  size_t lid = get_local_id(0);
  size_t localSize = get_local_size(0);

  ulong c = groupId;
  ulong spatialSize = H * W;
  ulong totalElems = N * spatialSize;

  TYPE sum = (TYPE)0;

  for (ulong idx = lid; idx < totalElems; idx += localSize) {
    ulong n = idx / spatialSize;
    ulong s = idx % spatialSize;
    ulong addr = n * sampleStride + actv_in_offset + c * spatialSize + s;
    sum += batch_actvs[addr];
  }

  partials[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (ulong stride = localSize / 2; stride > 0; stride >>= 1) {
    if (lid < stride)
      partials[lid] += partials[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    bn_batch_mean[norm_param_offset + c] = partials[0] / (TYPE)totalElems;
  }
}

//===================================================================================================================//

// Compute per-channel variance across ALL N samples: var_c = sum_{n,h,w} (x - mean)^2 / (N*H*W)
kernel void batchnorm_batch_compute_var(global TYPE* batch_actvs, global TYPE* bn_batch_mean, global TYPE* bn_batch_var,
                                        ulong actv_in_offset, ulong norm_param_offset, ulong C, ulong H, ulong W,
                                        ulong N, ulong sampleStride)
{
  local TYPE partials[256];

  size_t groupId = get_group_id(0);
  size_t lid = get_local_id(0);
  size_t localSize = get_local_size(0);

  ulong c = groupId;
  ulong spatialSize = H * W;
  ulong totalElems = N * spatialSize;
  TYPE mean = bn_batch_mean[norm_param_offset + c];

  TYPE sum = (TYPE)0;

  for (ulong idx = lid; idx < totalElems; idx += localSize) {
    ulong n = idx / spatialSize;
    ulong s = idx % spatialSize;
    ulong addr = n * sampleStride + actv_in_offset + c * spatialSize + s;
    TYPE diff = batch_actvs[addr] - mean;
    sum += diff * diff;
  }

  partials[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (ulong stride = localSize / 2; stride > 0; stride >>= 1) {
    if (lid < stride)
      partials[lid] += partials[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    bn_batch_var[norm_param_offset + c] = partials[0] / (TYPE)totalElems;
  }
}

//===================================================================================================================//

// Normalize ALL N samples using batch-wide mean/var, store xnorm, apply gamma/beta
// Global work size = N * C * H * W (all elements across all samples)
kernel void batchnorm_batch_normalize(global TYPE* batch_actvs, global TYPE* batch_xnorm, global TYPE* bn_gamma,
                                      global TYPE* bn_beta, global TYPE* bn_mean, global TYPE* bn_var,
                                      ulong actv_in_offset, ulong actv_out_offset, ulong xnorm_offset,
                                      ulong norm_param_offset, ulong C, ulong H, ulong W, ulong N, ulong sampleStride,
                                      float epsilon)
{
  size_t gid = get_global_id(0);
  ulong spatialSize = H * W;
  ulong sampleSize = C * spatialSize;
  ulong totalSize = N * sampleSize;

  if (gid >= totalSize)
    return;

  ulong n = gid / sampleSize;
  ulong withinSample = gid % sampleSize;
  ulong c = withinSample / spatialSize;

  ulong inAddr = n * sampleStride + actv_in_offset + withinSample;
  ulong outAddr = n * sampleStride + actv_out_offset + withinSample;
  ulong xnAddr = n * sampleStride + xnorm_offset + withinSample;

  TYPE x = batch_actvs[inAddr];
  TYPE mean = bn_mean[norm_param_offset + c];
  TYPE var = bn_var[norm_param_offset + c];
  TYPE gamma = bn_gamma[norm_param_offset + c];
  TYPE beta = bn_beta[norm_param_offset + c];

  TYPE invStd = (TYPE)1 / sqrt(var + (TYPE)epsilon);
  TYPE xn = (x - mean) * invStd;
  batch_xnorm[xnAddr] = xn;
  batch_actvs[outAddr] = gamma * xn + beta;
}

//===================================================================================================================//

// Compute dGamma and dBeta across ALL N samples
// One work-group per channel
kernel void batchnorm_batch_dGammaBeta(global TYPE* batch_grads, global TYPE* batch_xnorm, global TYPE* bn_dGamma,
                                       global TYPE* bn_dBeta, ulong grad_out_offset, ulong xnorm_offset,
                                       ulong norm_param_offset, ulong C, ulong H, ulong W, ulong N, ulong sampleStride)
{
  local TYPE partials_dg[256];
  local TYPE partials_db[256];

  size_t groupId = get_group_id(0);
  size_t lid = get_local_id(0);
  size_t localSize = get_local_size(0);

  ulong c = groupId;
  ulong spatialSize = H * W;
  ulong totalElems = N * spatialSize;

  TYPE sum_dg = (TYPE)0;
  TYPE sum_db = (TYPE)0;

  for (ulong idx = lid; idx < totalElems; idx += localSize) {
    ulong n = idx / spatialSize;
    ulong s = idx % spatialSize;
    ulong gradAddr = n * sampleStride + grad_out_offset + c * spatialSize + s;
    ulong xnAddr = n * sampleStride + xnorm_offset + c * spatialSize + s;
    TYPE dOut = batch_grads[gradAddr];
    TYPE xn = batch_xnorm[xnAddr];
    sum_dg += dOut * xn;
    sum_db += dOut;
  }

  partials_dg[lid] = sum_dg;
  partials_db[lid] = sum_db;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (ulong stride = localSize / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      partials_dg[lid] += partials_dg[lid + stride];
      partials_db[lid] += partials_db[lid + stride];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    bn_dGamma[norm_param_offset + c] = partials_dg[0];
    bn_dBeta[norm_param_offset + c] = partials_db[0];
  }
}

//===================================================================================================================//

// Compute dInput for ALL N samples using batch-wide stats
// Global work size = N * C * H * W
kernel void batchnorm_batch_dInput(global TYPE* batch_grads, global TYPE* batch_xnorm, global TYPE* bn_gamma,
                                   global TYPE* bn_dGamma, global TYPE* bn_dBeta, global TYPE* bn_batch_var,
                                   ulong grad_in_offset, ulong grad_out_offset, ulong xnorm_offset,
                                   ulong norm_param_offset, ulong C, ulong H, ulong W, ulong N, ulong sampleStride,
                                   float epsilon)
{
  size_t gid = get_global_id(0);
  ulong spatialSize = H * W;
  ulong sampleSize = C * spatialSize;
  ulong totalSize = N * sampleSize;

  if (gid >= totalSize)
    return;

  ulong n = gid / sampleSize;
  ulong withinSample = gid % sampleSize;
  ulong c = withinSample / spatialSize;

  ulong gradOutAddr = n * sampleStride + grad_out_offset + withinSample;
  ulong gradInAddr = n * sampleStride + grad_in_offset + withinSample;
  ulong xnAddr = n * sampleStride + xnorm_offset + withinSample;

  TYPE gamma = bn_gamma[norm_param_offset + c];
  TYPE var = bn_batch_var[norm_param_offset + c];
  TYPE invStd = (TYPE)1 / sqrt(var + (TYPE)epsilon);
  TYPE NHW = (TYPE)(N * spatialSize);
  TYPE dg = bn_dGamma[norm_param_offset + c];
  TYPE db = bn_dBeta[norm_param_offset + c];
  TYPE xn = batch_xnorm[xnAddr];
  TYPE dOut = batch_grads[gradOutAddr];

  batch_grads[gradInAddr] = (gamma * invStd / NHW) * (NHW * dOut - db - xn * dg);
}

//===================================================================================================================//

#endif // CNN_BATCHNORM_CPP_CL
