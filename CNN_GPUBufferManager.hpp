#ifndef CNN_GPUBUFFERMANAGER_HPP
#define CNN_GPUBUFFERMANAGER_HPP

#include "CNN_Types.hpp"
#include "CNN_CoreGPUWorkerConfig.hpp"
#include "CNN_Parameters.hpp"
#include "CNN_LogLevel.hpp"

#include <ANN_CoreGPUWorker.hpp>
#include <OCLW_Core.hpp>

#include <memory>
#include <stack>
#include <vector>

//===================================================================================================================//

namespace CNN
{
  // Manages CNN GPU buffer allocation, layer offset computation, source loading,
  // ANN worker construction, and data I/O. Extracted from CoreGPUWorker.
  template <typename T>
  class GPUBufferManager
  {
    public:
      GPUBufferManager(OpenCLWrapper::Core* core, const CoreGPUWorkerConfig<T>& workerConfig,
                       Parameters<T>& parameters);

      //-- Initialization --//
      void computeLayerOffsets();
      void loadSources(bool skipDefines);
      void allocateBuffers(ulong batchSize = 1);
      void buildANNWorker();

      //-- Parameter synchronization --//
      void syncParametersFromGPU(); // GPU → CPU (after training)
      void syncParametersToGPU(); // CPU → GPU (for validation with updated params)
      void setParameters(const Parameters<T>& params)
      {
        parameters = params;
      }

      //-- Accumulator operations --//
      void resetAccumulators();
      void readAccumulatedGradients(std::vector<T>& accumFilters, std::vector<T>& accumBiases);
      void setAccumulators(const std::vector<T>& accumFilters, const std::vector<T>& accumBiases);
      void readBNAccumulatedGradients(std::vector<T>& accumGamma, std::vector<T>& accumBeta);
      void setBNAccumulators(const std::vector<T>& accumGamma, const std::vector<T>& accumBeta);
      void readANNAccumulatedGradients(ANN::Tensor1D<T>& accumWeights, ANN::Tensor1D<T>& accumBiases);
      void setANNAccumulators(const ANN::Tensor1D<T>& accumWeights, const ANN::Tensor1D<T>& accumBiases);

      //-- Per-layer buffer offset/size info --//
      struct LayerInfo {
          ulong actvOffset;
          ulong actvSize;
      };

      struct ConvInfo {
          ulong filterOffset;
          ulong biasOffset;
          ulong numFilterElems;
          ulong numBiases;
      };

      struct PoolInfo {
          ulong indexOffset;
          ulong indexSize;
      };

      struct InstanceNormInfo {
          ulong paramOffset; // Offset into flat norm_gamma/norm_beta/norm_running_mean/norm_running_var buffers
          ulong numChannels;
      };

      //-- Public data (accessed by CoreGPUWorker and GPUKernelBuilder) --//
      std::vector<LayerInfo> layerInfos;
      ulong totalActvSize = 0;

      std::vector<ConvInfo> convInfos;
      ulong totalFilterSize = 0;
      ulong totalBiasSize = 0;

      std::vector<PoolInfo> poolInfos;
      ulong totalPoolIndexSize = 0;

      std::vector<InstanceNormInfo> normInfos;
      ulong totalNormParamSize = 0;

      struct ResidualProjInfo {
          ulong skipOffset; // Activation offset of the residual_start input
          ulong inC; // Input channels (0 = identity shortcut)
          ulong outC; // Output channels
          ulong spatialSize; // H * W
          ulong paramOffset; // Offset into flat residual param buffer
      };

      struct ResidualShapeInfo {
          Shape3D shape;
          ulong actvOffset;
      };

      std::vector<ResidualProjInfo> residualProjInfos;
      ulong totalResidualParamSize = 0;
      std::stack<ResidualShapeInfo> residualShapeStack; // Used during computeLayerOffsets

      Shape3D cnnOutputShape;
      ulong flattenSize = 0;

      //-- im2col workspace --//
      ulong maxIm2ColSize = 0; // max(C_in * kH * kW * outH * outW) across all conv layers

      //-- Batch size --//
      ulong batchSize = 1;

      //-- ANN GPU worker (dense layers on shared core) --//
      std::unique_ptr<ANN::CoreGPUWorker<T>> annGPUWorker;

    private:
      OpenCLWrapper::Core* core;
      const CoreGPUWorkerConfig<T>& workerConfig;
      Parameters<T>& parameters;
  };
}

//===================================================================================================================//

#endif // CNN_GPUBUFFERMANAGER_HPP
