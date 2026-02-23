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
      //-- Constructor --//
      CoreCPU(const CoreConfig<T>& config);

      //-- Core interface --//
      Output<T> predict(const Input<T>& input) override;
      void train(const Samples<T>& samples) override;
      TestResult<T> test(const Samples<T>& samples) override;

    private:
      //-- ANN sub-core --//
      std::unique_ptr<ANN::Core<T>> annCore;

      //-- CNN output shape --//
      Shape3D cnnOutputShape;
      ulong flattenSize;

      //-- Forward / Backward pass --//
      Tensor3D<T> forwardCNN(const Input<T>& input,
                             std::vector<Tensor3D<T>>& intermediates,
                             std::vector<std::vector<ulong>>& poolMaxIndices);
      Tensor3D<T> forwardCNN(const Input<T>& input);
      void backwardCNN(const Tensor3D<T>& dCNNOut,
                       const std::vector<Tensor3D<T>>& intermediates,
                       const std::vector<std::vector<ulong>>& poolMaxIndices,
                       std::vector<std::vector<T>>& dConvFilters,
                       std::vector<std::vector<T>>& dConvBiases);

      //-- Initialization --//
      ANN::CoreConfig<T> buildANNConfig(const CoreConfig<T>& cnnConfig);
      void initializeConvParams();

      //-- Loss calculation --//
      T calculateLoss(const Output<T>& predicted, const Output<T>& expected);

      //-- CNN gradient accumulation --//
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

