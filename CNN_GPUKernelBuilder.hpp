#ifndef CNN_GPUKERNELBUILDER_HPP
#define CNN_GPUKERNELBUILDER_HPP

#include "CNN_Types.hpp"
#include "CNN_CoreConfig.hpp"
#include "CNN_GPUBufferManager.hpp"
#include "CNN_LogLevel.hpp"

#include <OCLW_Core.hpp>

//===================================================================================================================//

namespace CNN
{
  // Manages OpenCL kernel creation, argument binding, and setup orchestration for CNN.
  // Extracted from CoreGPUWorker to reduce class size.
  template <typename T>
  class GPUKernelBuilder
  {
    public:
      GPUKernelBuilder(OpenCLWrapper::Core* core, const CoreConfig<T>& coreConfig, GPUBufferManager<T>& bufferManager,
                       LogLevel logLevel);

      //-- Kernel setup (clears previous kernels and rebuilds) --//
      void setupPredictKernels();
      void setupTrainingKernels();
      void setupUpdateKernels(ulong numSamples);

      //-- Kernel building blocks (public for external orchestration) --//
      void addPropagateKernels(bool training = false);
      void addCopyBridgeKernels();
      void addBackpropagateKernels();
      void addCNNAccumulateKernels();
      void addCNNUpdateKernels(ulong numSamples);

      //-- Batch-norm-aware kernel building blocks --//
      // Add propagation kernels for a single sample within batch buffers
      void addBatchPropagateKernelsForSample(ulong sampleIdx, ulong layerStart, ulong layerEnd);
      // Add batch-wide BN forward kernels (cross-sample reduction)
      void addBatchNormForwardKernels(ulong layerIdx, ulong batchSize);
      // Add backpropagation kernels for a single sample within batch buffers
      void addBatchBackpropagateKernelsForSample(ulong sampleIdx, ulong layerStart, ulong layerEnd);
      // Add batch-wide BN backward kernels (cross-sample reduction)
      void addBatchNormBackwardKernels(ulong layerIdx, ulong batchSize);
      // Add copy bridge from batch buffer to single-sample ANN buffer
      void addBatchCopyBridgeKernels(ulong sampleIdx);
      // Add reverse bridge from ANN grads to batch gradient buffer
      void addBatchReverseBridgeKernels(ulong sampleIdx);
      // Add accumulate kernels for batch-norm training (CNN only, from batch buffers)
      void addBatchCNNAccumulateKernelsForSample(ulong sampleIdx, ulong layerStart, ulong layerEnd);
      // Add batch-norm running stats update using batch-wide mean/var directly
      void addBatchNormRunningStatsUpdate(ulong batchSize);

      //-- Kernel setup flags --//
      bool predictKernelsSetup = false;
      bool trainingKernelsSetup = false;
      bool updateKernelsSetup = false;
      bool skipBNRunningStatsInUpdate = false; // Set by batch norm training path

      void invalidateAllKernelFlags();

    private:
      OpenCLWrapper::Core* core;
      const CoreConfig<T>& coreConfig;
      GPUBufferManager<T>& bufferManager;
      LogLevel logLevel;

      //-- Adam optimizer state --//
      ulong adam_t = 0;
  };
}

//===================================================================================================================//

#endif // CNN_GPUKERNELBUILDER_HPP
