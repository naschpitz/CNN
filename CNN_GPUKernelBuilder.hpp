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
      void addPropagateKernels();
      void addCopyBridgeKernels();
      void addBackpropagateKernels();
      void addCNNAccumulateKernels();
      void addCNNUpdateKernels(ulong numSamples);

      //-- Kernel setup flags --//
      bool predictKernelsSetup = false;
      bool trainingKernelsSetup = false;
      bool updateKernelsSetup = false;

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
