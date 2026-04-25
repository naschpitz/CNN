#ifndef CNN_GPUKERNELBUILDER_HPP
#define CNN_GPUKERNELBUILDER_HPP

#include "CNN_Types.hpp"
#include "CNN_CoreGPUWorkerConfig.hpp"
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
    GPUKernelBuilder(OpenCLWrapper::Core* core, const CoreGPUWorkerConfig<T>& workerConfig,
                     GPUBufferManager<T>& bufferManager);

    //-- Kernel setup (clears previous kernels and rebuilds) --//
    void setupPredictKernels();
    void setupTrainingKernels();
    void setupUpdateKernels(ulong numSamples);

    //-- Kernel building blocks (public for external orchestration) --//
    void addPropagateKernels(ulong sampleIdx, ulong layerStart, ulong layerEnd, bool training = false);
    void addBackpropagateKernels(ulong sampleIdx, ulong layerStart, ulong layerEnd);
    void addCopyBridgeKernels(ulong sampleIdx);
    void addReverseBridgeKernels(ulong sampleIdx);
    void addCNNAccumulateKernels(ulong sampleIdx, ulong layerStart, ulong layerEnd);
    void addCNNUpdateKernels(ulong numSamples, bool skipBNRunningStats = false);

    //-- Cross-sample batch normalization kernels --//
    void addBatchNormForwardKernels(ulong layerIdx, ulong batchSize);
    void addBatchNormBackwardKernels(ulong layerIdx, ulong batchSize);
    void addBatchNormRunningStatsUpdate(ulong batchSize);

    //-- Kernel setup flags --//
    bool predictKernelsSetup = false;
    bool trainingKernelsSetup = false;
    bool updateKernelsSetup = false;
    bool skipBNRunningStatsInUpdate = false; // Set by trainSubset when BN path is active

    void invalidateAllKernelFlags();

  private:
    OpenCLWrapper::Core* core;
    const CoreGPUWorkerConfig<T>& workerConfig;
    GPUBufferManager<T>& bufferManager;

    //-- Adam optimizer state --//
    ulong adam_t = 0;
  };
}

//===================================================================================================================//

#endif // CNN_GPUKERNELBUILDER_HPP
