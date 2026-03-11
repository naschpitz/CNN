#ifndef CNN_NORMALIZATION_CPP_CL
#define CNN_NORMALIZATION_CPP_CL

//===================================================================================================================//
// CNN_Normalization.cpp.cl
//
// Unified normalization kernels for both InstanceNorm and BatchNorm.
//
// All kernels accept N (number of samples) and sampleStride parameters:
//   - InstanceNorm / per-sample: N=1, sampleStride=0
//   - BatchNorm / cross-sample:  N=batchSize, sampleStride=<actual stride>
//===================================================================================================================//

//===================================================================================================================//
//-- Forward pass --//
//===================================================================================================================//

// Compute per-channel mean across N samples: mean_c = sum_{n,h,w} x[n,c,h,w] / (N*H*W)
// One work-group per channel. Uses tree reduction.
kernel void norm_compute_mean(global TYPE* actvs, global TYPE* norm_mean, ulong actv_in_offset, ulong norm_param_offset,
                              ulong C, ulong H, ulong W, ulong N, ulong sampleStride)
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
    sum += actvs[addr];
  }

  partials[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (ulong stride = localSize / 2; stride > 0; stride >>= 1) {
    if (lid < stride)
      partials[lid] += partials[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    norm_mean[norm_param_offset + c] = partials[0] / (TYPE)totalElems;
  }
}

//===================================================================================================================//

// Compute per-channel variance across N samples: var_c = sum_{n,h,w} (x - mean)^2 / (N*H*W)
// One work-group per channel. Uses tree reduction. Requires mean already computed.
kernel void norm_compute_var(global TYPE* actvs, global TYPE* norm_mean, global TYPE* norm_var, ulong actv_in_offset,
                             ulong norm_param_offset, ulong C, ulong H, ulong W, ulong N, ulong sampleStride)
{
  local TYPE partials[256];

  size_t groupId = get_group_id(0);
  size_t lid = get_local_id(0);
  size_t localSize = get_local_size(0);

  ulong c = groupId;
  ulong spatialSize = H * W;
  ulong totalElems = N * spatialSize;
  TYPE mean = norm_mean[norm_param_offset + c];

  TYPE sum = (TYPE)0;

  for (ulong idx = lid; idx < totalElems; idx += localSize) {
    ulong n = idx / spatialSize;
    ulong s = idx % spatialSize;
    ulong addr = n * sampleStride + actv_in_offset + c * spatialSize + s;
    TYPE diff = actvs[addr] - mean;
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
    norm_var[norm_param_offset + c] = partials[0] / (TYPE)totalElems;
  }
}

//===================================================================================================================//

// Normalize, scale, shift, and store xnorm for N samples.
// Global work size = N * C * H * W.
kernel void norm_normalize(global TYPE* actvs, global TYPE* norm_xnorm, global TYPE* norm_gamma, global TYPE* norm_beta,
                           global TYPE* norm_mean, global TYPE* norm_var, ulong actv_in_offset, ulong actv_out_offset,
                           ulong xnorm_offset, ulong norm_param_offset, ulong C, ulong H, ulong W, ulong N,
                           ulong sampleStride, float epsilon)
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

  TYPE x = actvs[inAddr];
  TYPE mean = norm_mean[norm_param_offset + c];
  TYPE var = norm_var[norm_param_offset + c];
  TYPE gamma = norm_gamma[norm_param_offset + c];
  TYPE beta = norm_beta[norm_param_offset + c];

  TYPE invStd = (TYPE)1 / sqrt(var + (TYPE)epsilon);
  TYPE xn = (x - mean) * invStd;
  norm_xnorm[xnAddr] = xn;
  actvs[outAddr] = gamma * xn + beta;
}

//===================================================================================================================//
//-- Backward pass --//
//===================================================================================================================//

// Compute dGamma and dBeta per channel across N samples (reduction).
// One work-group per channel.
kernel void norm_dGammaBeta(global TYPE* grads, global TYPE* norm_xnorm, global TYPE* norm_dGamma,
                            global TYPE* norm_dBeta, ulong grad_out_offset, ulong xnorm_offset, ulong norm_param_offset,
                            ulong C, ulong H, ulong W, ulong N, ulong sampleStride)
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
    TYPE dOut = grads[gradAddr];
    TYPE xn = norm_xnorm[xnAddr];
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
    norm_dGamma[norm_param_offset + c] = partials_dg[0];
    norm_dBeta[norm_param_offset + c] = partials_db[0];
  }
}

//===================================================================================================================//

// Compute dInput for N samples.
// Global work size = N * C * H * W.
kernel void norm_dInput(global TYPE* grads, global TYPE* norm_xnorm, global TYPE* norm_gamma, global TYPE* norm_dGamma,
                        global TYPE* norm_dBeta, global TYPE* norm_var, ulong grad_in_offset, ulong grad_out_offset,
                        ulong xnorm_offset, ulong norm_param_offset, ulong C, ulong H, ulong W, ulong N,
                        ulong sampleStride, float epsilon)
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

  TYPE gamma = norm_gamma[norm_param_offset + c];
  TYPE var = norm_var[norm_param_offset + c];
  TYPE invStd = (TYPE)1 / sqrt(var + (TYPE)epsilon);
  TYPE NHW = (TYPE)(N * spatialSize);
  TYPE dg = norm_dGamma[norm_param_offset + c];
  TYPE db = norm_dBeta[norm_param_offset + c];
  TYPE xn = norm_xnorm[xnAddr];
  TYPE dOut = grads[gradOutAddr];

  grads[gradInAddr] = (gamma * invStd / NHW) * (NHW * dOut - db - xn * dg);
}

//===================================================================================================================//
//-- Running stats update --//
//===================================================================================================================//

// Update running stats: running = (1 - momentum) * running + momentum * avgBatchStat
// One work-item per channel.
kernel void norm_update_running_stats(global TYPE* norm_running_mean, global TYPE* norm_running_var,
                                      global TYPE* accum_norm_batch_mean, global TYPE* accum_norm_batch_var,
                                      ulong norm_param_offset, ulong numChannels, float momentum, ulong numSamples)
{
  size_t gid = get_global_id(0);

  if (gid >= numChannels)
    return;

  ulong idx = norm_param_offset + gid;
  TYPE avgMean = accum_norm_batch_mean[idx] / (TYPE)numSamples;
  TYPE avgVar = accum_norm_batch_var[idx] / (TYPE)numSamples;
  norm_running_mean[idx] = ((TYPE)1 - (TYPE)momentum) * norm_running_mean[idx] + (TYPE)momentum * avgMean;
  norm_running_var[idx] = ((TYPE)1 - (TYPE)momentum) * norm_running_var[idx] + (TYPE)momentum * avgVar;
}

//===================================================================================================================//

#endif // CNN_NORMALIZATION_CPP_CL
