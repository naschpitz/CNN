#ifndef CNN_CORECPUWORKER_HPP
#define CNN_CORECPUWORKER_HPP

#include "CNN_Worker.hpp"
#include "CNN_Core.hpp"

#include <ANN_Core.hpp>

#include <memory>
#include <vector>

//===================================================================================================================//

namespace CNN {
  template <typename T>
  class CoreCPUWorker : public Worker<T> {
    public:
      CoreCPUWorker(const CoreConfig<T>& config,
                    const LayersConfig& layersConfig,
                    const Parameters<T>& sharedParams,
                    bool allocateTraining);

      //-- Predict (inference only — no intermediates saved) --//
      Output<T> predict(const Input<T>& input);

      //-- Full propagate+backpropagate+accumulate for one training sample --//
      T processSample(const Input<T>& input, const Output<T>& expected);

      //-- Accumulator management --//
      void resetAccumulators();

      //-- Loss accumulator --//
      T getAccumLoss() const { return accum_loss; }
      void resetAccumLoss() { accum_loss = static_cast<T>(0); }
      void addToAccumLoss(T loss) { accum_loss += loss; }

      //-- CNN gradient accumulator access (for merging by CoreCPU) --//
      const std::vector<std::vector<T>>& getAccumConvFilters() const { return accumDConvFilters; }
      const std::vector<std::vector<T>>& getAccumConvBiases() const { return accumDConvBiases; }

      //-- ANN sub-core access (for parameter sync/merge by CoreCPU) --//
      ANN::Core<T>* getANNCore() { return annCore.get(); }
      const ANN::Core<T>* getANNCore() const { return annCore.get(); }

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
      T accum_loss = static_cast<T>(0);

      //-- Propagate --//
      Tensor3D<T> propagateCNN(const Input<T>& input);
      Tensor3D<T> propagateCNN(const Input<T>& input,
                               std::vector<Tensor3D<T>>& intermediates,
                               std::vector<std::vector<ulong>>& poolMaxIndices);

      //-- Backpropagate --//
      void backpropagateCNN(const Tensor3D<T>& dCNNOut,
                       const std::vector<Tensor3D<T>>& intermediates,
                       const std::vector<std::vector<ulong>>& poolMaxIndices,
                       std::vector<std::vector<T>>& dConvFilters,
                       std::vector<std::vector<T>>& dConvBiases);

      //-- Initialization --//
      static ANN::CoreConfig<T> buildANNConfig(const CoreConfig<T>& cnnConfig, ulong flattenSize);
  };
}

//===================================================================================================================//

#endif // CNN_CORECPUWORKER_HPP

