#ifndef CNN_CORECPU_HPP
#define CNN_CORECPU_HPP

#include "CNN_Core.hpp"

#include <ANN_Core.hpp>

#include <memory>

//===================================================================================================================//

namespace CNN {
  template <typename T>
  class CoreCPU : public Core<T> {
    public:
      CoreCPU(const CoreConfig<T>& config);

      Output<T> predict(const Input<T>& input) override;
      void train(const Samples<T>& samples) override;
      TestResult<T> test(const Samples<T>& samples) override;

    private:
      std::unique_ptr<ANN::Core<T>> annCore;

      // Shape of the CNN output (before flatten)
      Shape3D cnnOutputShape;
      ulong flattenSize;

      // Forward pass through CNN layers only (no flatten/dense)
      // intermediates: stores input to each layer (for backprop)
      // poolMaxIndices: stores max indices for each pool layer
      Tensor3D<T> forwardCNN(const Input<T>& input,
                             std::vector<Tensor3D<T>>& intermediates,
                             std::vector<std::vector<ulong>>& poolMaxIndices);

      // Simplified forward (no intermediates stored - for predict-only)
      Tensor3D<T> forwardCNN(const Input<T>& input);

      // Backward pass through CNN layers
      void backwardCNN(const Tensor3D<T>& dCNNOut,
                       const std::vector<Tensor3D<T>>& intermediates,
                       const std::vector<std::vector<ulong>>& poolMaxIndices,
                       std::vector<std::vector<T>>& dConvFilters,
                       std::vector<std::vector<T>>& dConvBiases);

      // Build ANN core config from CNN config
      ANN::CoreConfig<T> buildANNConfig(const CoreConfig<T>& cnnConfig);

      // Initialize conv parameters with He initialization
      void initializeConvParams();

      // Calculate MSE loss
      T calculateLoss(const Output<T>& predicted, const Output<T>& expected);

      // Accumulated CNN gradients
      std::vector<std::vector<T>> accumDConvFilters;
      std::vector<std::vector<T>> accumDConvBiases;

      void resetCNNAccumulators();
      void accumulateCNNGradients(const std::vector<std::vector<T>>& dConvFilters,
                                  const std::vector<std::vector<T>>& dConvBiases);
      void updateCNNParameters(ulong numSamples);
  };
}

//===================================================================================================================//

#endif // CNN_CORECPU_HPP

