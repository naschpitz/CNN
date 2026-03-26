#ifndef CNN_DEFINES_HPP_CL
#define CNN_DEFINES_HPP_CL

#define TYPE float
#define TILE_SIZE 16 // Work-group tile dimension for GEMM kernels (16×16 = 256 work-items per group)

//===================================================================================================================//
// ANN types (must match C++ ANN::ActvFuncType and ANN::Layer)
// Defined here so they are available when ANN kernels are loaded with skipDefines=true.
//===================================================================================================================//

typedef enum { ACTV_RELU = 0, ACTV_SIGMOID = 1, ACTV_TANH = 2, ACTV_SOFTMAX = 3, ACTV_UNKNOWN = 4 } ActvFuncType;

typedef struct {
    ulong numNeurons;
    ActvFuncType actvFuncType;
} Layer;

#endif // CNN_DEFINES_HPP_CL
