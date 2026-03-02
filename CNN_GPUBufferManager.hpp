#ifndef CNN_GPUBUFFERMANAGER_HPP
#define CNN_GPUBUFFERMANAGER_HPP

#include "CNN_Types.hpp"
#include "CNN_CoreConfig.hpp"
#include "CNN_Parameters.hpp"
#include "CNN_LogLevel.hpp"

#include <ANN_CoreGPUWorker.hpp>
#include <OCLW_Core.hpp>

#include <memory>
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
      GPUBufferManager(OpenCLWrapper::Core* core, const CoreConfig<T>& coreConfig, Parameters<T>& parameters,
                       LogLevel logLevel);

      //-- Initialization --//
      void computeLayerOffsets();
      void loadSources(bool skipDefines);
      void allocateBuffers();
      void buildANNWorker();

      //-- Parameter synchronization: GPU → CPU --//
      void syncParametersFromGPU();

      //-- Accumulator operations --//
      void resetAccumulators();
      void readAccumulatedGradients(std::vector<T>& accumFilters, std::vector<T>& accumBiases);
      void setAccumulators(const std::vector<T>& accumFilters, const std::vector<T>& accumBiases);
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

      //-- Public data (accessed by CoreGPUWorker and GPUKernelBuilder) --//
      std::vector<LayerInfo> layerInfos;
      ulong totalActvSize = 0;

      std::vector<ConvInfo> convInfos;
      ulong totalFilterSize = 0;
      ulong totalBiasSize = 0;

      std::vector<PoolInfo> poolInfos;
      ulong totalPoolIndexSize = 0;

      Shape3D cnnOutputShape;
      ulong flattenSize = 0;

      //-- ANN GPU worker (dense layers on shared core) --//
      std::unique_ptr<ANN::CoreGPUWorker<T>> annGPUWorker;

    private:
      OpenCLWrapper::Core* core;
      const CoreConfig<T>& coreConfig;
      Parameters<T>& parameters;
      LogLevel logLevel;
  };
}

//===================================================================================================================//

#endif // CNN_GPUBUFFERMANAGER_HPP
