#ifndef CNN_BRIDGE_CPP_CL
#define CNN_BRIDGE_CPP_CL

// Note: Depends on CNN_Defines.hpp.cl (TYPE)

//===================================================================================================================//
// Bridge kernels: copy data between CNN and ANN buffers on the same OpenCL core.
// copy_cnn_to_ann: copies the last CNN activation (flatten output) into ANN's actvs buffer (layer 0).
// copy_ann_grad_to_cnn: copies ANN's input gradients (dCost_dActvs, layer 0) into CNN's gradient buffer.
//===================================================================================================================//

__kernel void copy_cnn_to_ann(__global const TYPE* cnn_actvs, __global TYPE* actvs, const ulong cnn_offset,
                              const ulong size)
{
  ulong gid = get_global_id(0);

  if (gid >= size)
    return;

  actvs[gid] = cnn_actvs[cnn_offset + gid];
}

//===================================================================================================================//

__kernel void copy_ann_grad_to_cnn(__global const TYPE* dCost_dActvs, __global TYPE* cnn_grads, const ulong cnn_offset,
                                   const ulong size)
{
  ulong gid = get_global_id(0);

  if (gid >= size)
    return;

  cnn_grads[cnn_offset + gid] = dCost_dActvs[gid];
}

//===================================================================================================================//

#endif // CNN_BRIDGE_CPP_CL
