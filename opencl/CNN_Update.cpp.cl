#ifndef CNN_UPDATE_CPP_CL
#define CNN_UPDATE_CPP_CL

// Note: Depends on CNN_Defines.hpp.cl (TYPE)

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

// Adam optimizer update for CNN parameters
kernel void update_parameters_adam(
    global TYPE* params,
    global TYPE* accum,
    global TYPE* m,
    global TYPE* v,
    ulong offset,
    ulong size,
    ulong numSamples,
    float learningRate,
    float beta1,
    float beta2,
    float epsilon,
    float bc1,
    float bc2
  ) {
  size_t gid = get_global_id(0);
  if (gid >= size) return;

  TYPE g = accum[offset + gid] / (TYPE)numSamples;
  m[offset + gid] = beta1 * m[offset + gid] + (1.0f - beta1) * g;
  v[offset + gid] = beta2 * v[offset + gid] + (1.0f - beta2) * g * g;
  TYPE m_hat = m[offset + gid] / bc1;
  TYPE v_hat = v[offset + gid] / bc2;
  params[offset + gid] -= learningRate * m_hat / (sqrt(v_hat) + epsilon);
}

//===================================================================================================================//

#endif // CNN_UPDATE_CPP_CL

