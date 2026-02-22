#ifndef CNN_COREGPUWORKER_HPP
#define CNN_COREGPUWORKER_HPP

#include "CNN_Core.hpp"

#include <ANN_CoreGPUWorker.hpp>
#include <OCLW_Core.hpp>

#include <memory>

//===================================================================================================================//

namespace CNN {
  template <typename T>
  class CoreGPUWorker {
    public:
      // Standalone constructor — creates its own OpenCL core
      CoreGPUWorker(const CoreConfig<T>& config);

      // Shared-core constructor — uses externally-provided OpenCL core.
      // Only initializes parameters and computes offsets. Caller must invoke
      // loadSources(), allocateBuffers() manually.
      CoreGPUWorker(const CoreConfig<T>& config, OpenCLWrapper::Core& sharedCore);

      //-- Predict --//
      Output<T> predict(const Input<T>& input);

      //-- Training (called by CoreGPU orchestrator) --//
      T trainSubset(const Samples<T>& samples, ulong startIdx, ulong endIdx,
                    ulong epoch, ulong totalEpochs, const TrainingCallback<T>& callback);

      //-- Testing --//
      T testSubset(const Samples<T>& samples, ulong startIdx, ulong endIdx);

      //-- Step-by-step training methods (for external orchestration) --//
      void backpropagateSample(const Input<T>& input, const Output<T>& expected);
      void accumulate();
      void resetAccumulators();

      //-- CNN gradient access (for multi-GPU merging) --//
      void readAccumulatedGradients(std::vector<T>& accumFilters, std::vector<T>& accumBiases);
      void setAccumulators(const std::vector<T>& accumFilters, const std::vector<T>& accumBiases);

      //-- ANN gradient access (for multi-GPU merging) --//
      void readANNAccumulatedGradients(ANN::Tensor1D<T>& accumWeights, ANN::Tensor1D<T>& accumBiases);
      void setANNAccumulators(const ANN::Tensor1D<T>& accumWeights, const ANN::Tensor1D<T>& accumBiases);

      //-- Weight update --//
      void update(ulong numSamples);

      //-- Parameter synchronization --//
      void syncParametersFromGPU();

      //-- Parameter access --//
      const Parameters<T>& getParameters() const { return parameters; }

      //-- Shared-core integration: source loading and buffer allocation --//
      void loadSources(bool skipDefines);
      void allocateBuffers();

      //-- Shared-core integration: kernel building blocks --//
      void addForwardKernels();
      void addCopyBridgeKernels();
      void addBackwardKernels();
      void addCNNAccumulateKernels();
      void addCNNUpdateKernels(ulong numSamples);

    private:
      //-- Configuration --//
      CoreConfig<T> coreConfig;
      Parameters<T> parameters;
      bool verbose = false;

      //-- CNN shape info --//
      Shape3D cnnOutputShape;
      ulong flattenSize;

      //-- Per-layer buffer offset/size info --//
      struct LayerInfo {
        ulong actvOffset;  // offset in cnn_actvs / cnn_grads
        ulong actvSize;    // number of elements for this layer's output
      };
      std::vector<LayerInfo> layerInfos;  // index 0 = input, 1..N = CNN layer outputs
      ulong totalActvSize = 0;

      // Conv layer offsets into flat filter/bias buffers
      struct ConvInfo {
        ulong filterOffset;
        ulong biasOffset;
        ulong numFilterElems;
        ulong numBiases;
      };
      std::vector<ConvInfo> convInfos;
      ulong totalFilterSize = 0;
      ulong totalBiasSize = 0;

      // Pool layer indices offset
      struct PoolInfo {
        ulong indexOffset;
        ulong indexSize;
      };
      std::vector<PoolInfo> poolInfos;
      ulong totalPoolIndexSize = 0;

      //-- OpenCL state --//
      std::unique_ptr<OpenCLWrapper::Core> ownedCore;  // Owned core (standalone mode)
      OpenCLWrapper::Core* core = nullptr;              // Pointer to active core (owned or shared)

      //-- ANN GPU worker (dense layers on shared core) --//
      std::unique_ptr<ANN::CoreGPUWorker<T>> annGPUWorker;

      //-- Kernel setup flags --//
      bool predictKernelsSetup = false;
      bool trainingKernelsSetup = false;
      bool updateKernelsSetup = false;

      //-- Initialization --//
      void computeLayerOffsets();
      void initializeConvParams();
      void buildANNWorker();

      //-- Kernel setup (standalone mode) --//
      void setupPredictKernels();
      void setupTrainingKernels();
      void setupUpdateKernels(ulong numSamples);

      //-- Helpers --//
      void invalidateAllKernelFlags();
      T calculateLoss(const Output<T>& predicted, const Output<T>& expected);
  };
}

//===================================================================================================================//

#endif // CNN_COREGPUWORKER_HPP

