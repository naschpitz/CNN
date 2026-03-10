#ifndef CNN_INSTANCENORM_CPP_CL
#define CNN_INSTANCENORM_CPP_CL

// Note: Depends on CNN_Defines.hpp.cl (TYPE)

//===================================================================================================================//

// Batch norm forward pass (training): compute per-channel mean over spatial dims
// One work-group per channel. Uses tree reduction.
kernel void calculate_norm_mean(global TYPE* actvs, global TYPE* norm_batch_mean, ulong actv_in_offset,
                                ulong norm_param_offset, ulong C, ulong H, ulong W)
{
  local TYPE partials[256];

  size_t groupId = get_group_id(0); // channel index
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
    norm_batch_mean[norm_param_offset + c] = partials[0] / (TYPE)spatialSize;
  }
}

//===================================================================================================================//

// Batch norm forward pass (training): compute per-channel variance over spatial dims
// One work-group per channel. Uses tree reduction. Requires mean already computed.
kernel void calculate_norm_var(global TYPE* actvs, global TYPE* norm_batch_mean, global TYPE* norm_batch_var,
                               ulong actv_in_offset, ulong norm_param_offset, ulong C, ulong H, ulong W)
{
  local TYPE partials[256];

  size_t groupId = get_group_id(0);
  size_t lid = get_local_id(0);
  size_t localSize = get_local_size(0);

  ulong c = groupId;
  ulong spatialSize = H * W;
  ulong base = actv_in_offset + c * spatialSize;
  TYPE mean = norm_batch_mean[norm_param_offset + c];

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
    norm_batch_var[norm_param_offset + c] = partials[0] / (TYPE)spatialSize;
  }
}

//===================================================================================================================//

// Batch norm forward pass: normalize, scale, shift, and store xnorm
// Used for both training and inference. The mean/var buffers point to either
// batch stats (training, computed by mean/var kernels) or running stats (inference).
// One work-item per element. nElements = C * H * W
kernel void calculate_norm_normalize(global TYPE* actvs, global TYPE* norm_xnorm, global TYPE* norm_gamma,
                                     global TYPE* norm_beta, global TYPE* bn_mean, global TYPE* bn_var,
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
  TYPE mean = bn_mean[norm_param_offset + c];
  TYPE var = bn_var[norm_param_offset + c];
  TYPE gamma = norm_gamma[norm_param_offset + c];
  TYPE beta = norm_beta[norm_param_offset + c];

  TYPE invStd = (TYPE)1 / sqrt(var + (TYPE)epsilon);
  TYPE xn = (x - mean) * invStd;
  norm_xnorm[xnorm_offset + gid] = xn;
  actvs[actv_out_offset + gid] = gamma * xn + beta;
}

//===================================================================================================================//

// Batch norm backward: compute dGamma and dBeta per channel (reduction)
// One work-group per channel.
kernel void calculate_norm_dGammaBeta(global TYPE* grads, global TYPE* norm_xnorm, global TYPE* norm_dGamma,
                                      global TYPE* norm_dBeta, ulong grad_out_offset, ulong xnorm_offset,
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
    TYPE xn = norm_xnorm[xnorm_offset + idx];
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

// Batch norm backward: compute dInput
// One work-item per element. nElements = C * H * W
kernel void calculate_norm_dInput(global TYPE* grads, global TYPE* norm_xnorm, global TYPE* norm_gamma,
                                  global TYPE* norm_dGamma, global TYPE* norm_dBeta, global TYPE* norm_batch_var,
                                  ulong grad_in_offset, ulong grad_out_offset, ulong xnorm_offset,
                                  ulong norm_param_offset, ulong C, ulong H, ulong W, float epsilon)
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
  TYPE N = (TYPE)spatialSize;
  TYPE dg = norm_dGamma[norm_param_offset + c];
  TYPE db = norm_dBeta[norm_param_offset + c];
  TYPE xn = norm_xnorm[xnorm_offset + gid];
  TYPE dOut = grads[grad_out_offset + gid];

  grads[grad_in_offset + gid] = (gamma * invStd / N) * (N * dOut - db - xn * dg);
}

//===================================================================================================================//

// Update running stats: running = (1 - momentum) * running + momentum * batch
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

#endif // CNN_INSTANCENORM_CPP_CL
