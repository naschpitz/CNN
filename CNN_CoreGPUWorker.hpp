#ifndef CNN_COREGPUWORKER_HPP
#define CNN_COREGPUWORKER_HPP

#include "CNN_Core.hpp"

#include <ANN_Core.hpp>
#include <OCLW_Core.hpp>

#include <memory>

//===================================================================================================================//

namespace CNN {
  template <typename T>
  class CoreGPUWorker {
    public:
      CoreGPUWorker(const CoreConfig<T>& config);

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

      //-- Gradient access (for multi-GPU merging) --//
      void readAccumulatedGradients(std::vector<T>& accumFilters, std::vector<T>& accumBiases);
      void setAccumulators(const std::vector<T>& accumFilters, const std::vector<T>& accumBiases);

      //-- Weight update --//
      void update(ulong numSamples);
      void updateCNN(ulong numSamples);
      void updateANN(ulong numSamples);

      //-- Parameter synchronization --//
      void syncParametersFromGPU();

      //-- Parameter access --//
      const Parameters<T>& getParameters() const { return parameters; }
      ANN::Parameters<T> getANNParameters() const;
      void setANNParameters(const ANN::Parameters<T>& params);

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
      OpenCLWrapper::Core oclwCore;

      //-- ANN core for dense layers (CPU mode) --//
      std::unique_ptr<ANN::Core<T>> annCore;

      //-- Kernel setup flags --//
      bool predictKernelsSetup = false;
      bool backpropagateKernelsSetup = false;
      bool accumulateKernelsSetup = false;
      bool updateKernelsSetup = false;

      //-- Initialization --//
      void computeLayerOffsets();
      void initializeConvParams();
      void allocateBuffers();
      void buildANNCore();

      //-- Kernel setup --//
      void setupPredictKernels();
      void setupBackpropagateKernels();
      void setupAccumulateKernels();
      void setupUpdateKernels(ulong numSamples);

      //-- Kernel building blocks --//
      void addForwardKernels();
      void addBackwardKernels();
      void addAccumulateKernels();

      //-- Helpers --//
      void invalidateAllKernelFlags();
      T calculateLoss(const Output<T>& predicted, const Output<T>& expected);
      ANN::CoreConfig<T> buildANNConfig();
  };
}

//===================================================================================================================//

#endif // CNN_COREGPUWORKER_HPP

