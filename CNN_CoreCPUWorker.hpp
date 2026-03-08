#ifndef CNN_CORECPUWORKER_HPP
#define CNN_CORECPUWORKER_HPP

#include "CNN_Worker.hpp"
#include "CNN_Core.hpp"

#include <ANN_Core.hpp>

#include <memory>
#include <vector>

//===================================================================================================================//

namespace CNN
{
  template <typename T>
  class CoreCPUWorker : public Worker<T>
  {
    public:
      CoreCPUWorker(const CoreConfig<T>& config, const LayersConfig& layersConfig, const Parameters<T>& sharedParams,
                    bool allocateTraining);

      //-- Predict (inference only — no intermediates saved) --//
      Output<T> predict(const Input<T>& input);

      //-- Full propagate+backpropagate+accumulate for one training sample --//
      T processSample(const Input<T>& input, const Output<T>& expected);

      //-- Full batch processing with true BatchNorm: layer-by-layer forward/backward --//
      T processBatch(const std::vector<std::pair<Input<T>, Output<T>>>& batch);

      //-- Accumulator management --//
      void resetAccumulators();

      //-- Loss accumulator --//
      T getAccumLoss() const
      {
        return accum_loss;
      }

      void resetAccumLoss()
      {
        accum_loss = static_cast<T>(0);
      }

      void addToAccumLoss(T loss)
      {
        accum_loss += loss;
      }

      //-- CNN gradient accumulator access (for merging by CoreCPU) --//
      const std::vector<std::vector<T>>& getAccumConvFilters() const
      {
        return accumDConvFilters;
      }

      const std::vector<std::vector<T>>& getAccumConvBiases() const
      {
        return accumDConvBiases;
      }

      const std::vector<std::vector<T>>& getAccumNormGamma() const
      {
        return accumDNormGamma;
      }

      const std::vector<std::vector<T>>& getAccumNormBeta() const
      {
        return accumDNormBeta;
      }

      const std::vector<std::vector<T>>& getAccumNormMean() const
      {
        return accumNormMean;
      }

      const std::vector<std::vector<T>>& getAccumNormVar() const
      {
        return accumNormVar;
      }

      ulong getNormSampleCount() const
      {
        return normSampleCount;
      }

      //-- ANN sub-core access (for parameter sync/merge by CoreCPU) --//
      ANN::Core<T>* getANNCore()
      {
        return annCore.get();
      }

      const ANN::Core<T>* getANNCore() const
      {
        return annCore.get();
      }

    private:
      //-- Shared references (owned by CoreCPU/Core, read-only during propagate/backpropagate) --//
      const LayersConfig& layersConfig;
      const Parameters<T>& sharedParams;

      //-- CNN output shape --//
      Shape3D cnnOutputShape;
      ulong flattenSize;

      //-- ANN sub-core (each worker owns its own for thread safety) --//
      std::unique_ptr<ANN::Core<T>> annCore;

      //-- Per-worker CNN gradient accumulators --//
      std::vector<std::vector<T>> accumDConvFilters;
      std::vector<std::vector<T>> accumDConvBiases;
      std::vector<std::vector<T>> accumDNormGamma;
      std::vector<std::vector<T>> accumDNormBeta;
      std::vector<std::vector<T>> accumNormMean;
      std::vector<std::vector<T>> accumNormVar;
      ulong normSampleCount = 0;
      T accum_loss = static_cast<T>(0);

      //-- Per-worker norm training intermediates --//
      std::vector<std::vector<T>> normBatchMeans; // [normLayerIdx][channel]
      std::vector<std::vector<T>> normBatchVars; // [normLayerIdx][channel]
      std::vector<Tensor3D<T>> normXNormalized; // [normLayerIdx] (single-sample intermediates)

      //-- Per-worker true batch norm intermediates (batch-wide, for processBatch) --//
      std::vector<std::vector<T>> trueBNMeans; // [normLayerIdx][channel]
      std::vector<std::vector<T>> trueBNVars; // [normLayerIdx][channel]
      std::vector<std::vector<Tensor3D<T>>> trueBNXNorm; // [normLayerIdx][sampleIdx]

      //-- Propagate --//
      Tensor3D<T> propagateCNN(const Input<T>& input, bool training = false,
                               std::vector<Tensor3D<T>>* intermediates = nullptr,
                               std::vector<std::vector<ulong>>* poolMaxIndices = nullptr);

      //-- Backpropagate --//
      void backpropagateCNN(const Tensor3D<T>& dCNNOut, const std::vector<Tensor3D<T>>& intermediates,
                            const std::vector<std::vector<ulong>>& poolMaxIndices,
                            std::vector<std::vector<T>>& dConvFilters, std::vector<std::vector<T>>& dConvBiases,
                            std::vector<std::vector<T>>& dNormGamma, std::vector<std::vector<T>>& dNormBeta);

      //-- Initialization --//
      static ANN::CoreConfig<T> buildANNConfig(const CoreConfig<T>& cnnConfig, ulong flattenSize);
  };
}

//===================================================================================================================//

#endif // CNN_CORECPUWORKER_HPP
