#ifndef CNN_BATCHNORM_CPP_CL
#define CNN_BATCHNORM_CPP_CL

// Note: Depends on CNN_Defines.hpp.cl (TYPE)
//
// All normalization kernels are unified for both BatchNorm and InstanceNorm.
// - BatchNorm: N = batch size, sample_stride = per-sample buffer stride
// - InstanceNorm: N = 1, sample_stride = 0, actv_layer_offset = actv_in_offset

//===================================================================================================================//

// Normalization forward: compute per-channel mean across N×H×W.
// One work-group per channel. Uses tree reduction.
kernel void calculate_norm_mean(global TYPE* actvs, global TYPE* norm_batch_mean, ulong norm_param_offset, ulong C,
                                ulong H, ulong W, ulong N, ulong sample_stride, ulong actv_layer_offset)
{
  local TYPE partials[256];

  size_t groupId = get_group_id(0); // channel index
  size_t lid = get_local_id(0);
  size_t localSize = get_local_size(0);

  ulong c = groupId;
  ulong spatialSize = H * W;
  ulong totalElems = N * spatialSize;

  TYPE sum = (TYPE)0;

  for (ulong i = lid; i < totalElems; i += localSize) {
    ulong sampleIdx = i / spatialSize;
    ulong spatialIdx = i % spatialSize;
    ulong addr = sampleIdx * sample_stride + actv_layer_offset + c * spatialSize + spatialIdx;
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
    norm_batch_mean[norm_param_offset + c] = partials[0] / (TYPE)totalElems;
  }
}

//===================================================================================================================//

// Normalization forward: compute per-channel variance across N×H×W.
// One work-group per channel. Requires mean already computed.
kernel void calculate_norm_var(global TYPE* actvs, global TYPE* norm_batch_mean, global TYPE* norm_batch_var,
                               ulong norm_param_offset, ulong C, ulong H, ulong W, ulong N, ulong sample_stride,
                               ulong actv_layer_offset)
{
  local TYPE partials[256];

  size_t groupId = get_group_id(0);
  size_t lid = get_local_id(0);
  size_t localSize = get_local_size(0);

  ulong c = groupId;
  ulong spatialSize = H * W;
  ulong totalElems = N * spatialSize;
  TYPE mean = norm_batch_mean[norm_param_offset + c];

  TYPE sum = (TYPE)0;

  for (ulong i = lid; i < totalElems; i += localSize) {
    ulong sampleIdx = i / spatialSize;
    ulong spatialIdx = i % spatialSize;
    ulong addr = sampleIdx * sample_stride + actv_layer_offset + c * spatialSize + spatialIdx;
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
    norm_batch_var[norm_param_offset + c] = partials[0] / (TYPE)totalElems;
  }
}

//===================================================================================================================//

// Normalization forward: normalize, scale, shift, and store xnorm.
// Per-element kernel. nElements = C * H * W. Called once per sample.
kernel void calculate_norm_normalize(global TYPE* actvs, global TYPE* norm_xnorm, global TYPE* norm_gamma,
                                     global TYPE* norm_beta, global TYPE* norm_mean, global TYPE* norm_var,
                                     ulong actv_in_offset, ulong actv_out_offset, ulong xnorm_offset,
                                     ulong norm_param_offset, ulong C, ulong H, ulong W, float epsilon)
{
  size_t gid = get_global_id(0);
  ulong spatialSize = H * W;
  ulong totalSize = C * spatialSize;

  if (gid >= totalSize)
    return;

  ulong c = gid / spatialSize;
  TYPE x = actvs[actv_in_offset + gid];
  TYPE mean = norm_mean[norm_param_offset + c];
  TYPE var = norm_var[norm_param_offset + c];
  TYPE gamma = norm_gamma[norm_param_offset + c];
  TYPE beta = norm_beta[norm_param_offset + c];

  TYPE invStd = (TYPE)1 / sqrt(var + (TYPE)epsilon);
  TYPE xn = (x - mean) * invStd;
  norm_xnorm[xnorm_offset + gid] = xn;
  actvs[actv_out_offset + gid] = gamma * xn + beta;
}

//===================================================================================================================//

// Normalization backward: compute dGamma and dBeta per channel across N×H×W (reduction).
// One work-group per channel.
kernel void calculate_norm_dGammaBeta(global TYPE* grads, global TYPE* norm_xnorm, global TYPE* norm_dGamma,
                                      global TYPE* norm_dBeta, ulong norm_param_offset, ulong C, ulong H, ulong W,
                                      ulong N, ulong sample_stride, ulong grad_layer_offset, ulong xnorm_layer_offset)
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

  for (ulong i = lid; i < totalElems; i += localSize) {
    ulong sampleIdx = i / spatialSize;
    ulong spatialIdx = i % spatialSize;
    ulong gradAddr = sampleIdx * sample_stride + grad_layer_offset + c * spatialSize + spatialIdx;
    ulong xnormAddr = sampleIdx * sample_stride + xnorm_layer_offset + c * spatialSize + spatialIdx;
    TYPE dOut = grads[gradAddr];
    TYPE xn = norm_xnorm[xnormAddr];
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

// Normalization backward: compute dInput for one sample.
// M = total elements used in the mean/var computation (H*W for InstanceNorm, N*H*W for BatchNorm).
// Per-element kernel. nElements = C * H * W. Called once per sample.
kernel void calculate_norm_dInput(global TYPE* grads, global TYPE* norm_xnorm, global TYPE* norm_gamma,
                                  global TYPE* norm_dGamma, global TYPE* norm_dBeta, global TYPE* norm_batch_var,
                                  ulong grad_in_offset, ulong grad_out_offset, ulong xnorm_offset,
                                  ulong norm_param_offset, ulong C, ulong H, ulong W, float epsilon, ulong M)
{
  size_t gid = get_global_id(0);
  ulong spatialSize = H * W;
  ulong totalSize = C * spatialSize;

  if (gid >= totalSize)
    return;

  ulong c = gid / spatialSize;
  TYPE gamma = norm_gamma[norm_param_offset + c];
  TYPE var = norm_batch_var[norm_param_offset + c];
  TYPE invStd = (TYPE)1 / sqrt(var + (TYPE)epsilon);
  TYPE Mf = (TYPE)M;
  TYPE dg = norm_dGamma[norm_param_offset + c];
  TYPE db = norm_dBeta[norm_param_offset + c];
  TYPE xn = norm_xnorm[xnorm_offset + gid];
  TYPE dOut = grads[grad_out_offset + gid];

  grads[grad_in_offset + gid] = (gamma * invStd / Mf) * (Mf * dOut - db - xn * dg);
}

//===================================================================================================================//

// Update running stats directly from batch statistics (EMA):
//   running = (1 - momentum) * running + momentum * batch_stat
// Called during forward pass: once per batch for BatchNorm, once per sample for InstanceNorm.
// One work-item per channel.
kernel void update_norm_running_stats(global TYPE* norm_running_mean, global TYPE* norm_running_var,
                                      global TYPE* norm_batch_mean, global TYPE* norm_batch_var,
                                      ulong norm_param_offset, ulong numChannels, float momentum)
{
  size_t gid = get_global_id(0);

  if (gid >= numChannels)
    return;

  ulong idx = norm_param_offset + gid;
  norm_running_mean[idx] = ((TYPE)1 - (TYPE)momentum) * norm_running_mean[idx] + (TYPE)momentum * norm_batch_mean[idx];
  norm_running_var[idx] = ((TYPE)1 - (TYPE)momentum) * norm_running_var[idx] + (TYPE)momentum * norm_batch_var[idx];
}

//===================================================================================================================//

#endif // CNN_BATCHNORM_CPP_CL
