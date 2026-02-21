#ifndef CNN_CONV2D_HPP
#define CNN_CONV2D_HPP

#include "CNN_Types.hpp"
#include "CNN_Parameters.hpp"
#include "CNN_LayersConfig.hpp"

//===================================================================================================================//

namespace CNN {
  template <typename T>
  class Conv2D {
    public:
      // Forward pass: input -> output
      static Tensor3D<T> predict(const Tensor3D<T>& input, const ConvLayerConfig& config,
                                 const ConvParameters<T>& params);

      // Backpropagation: compute gradients
      // dOut: gradient of loss w.r.t. output of this layer [numFilters x outH x outW]
      // input: the input that was passed to predict()
      // config: the conv layer config
      // params: the conv layer parameters (filters, biases)
      // dFilters: [out] gradient w.r.t. filters (same shape as params.filters)
      // dBiases: [out] gradient w.r.t. biases (same shape as params.biases)
      // Returns: gradient of loss w.r.t. input [inputC x inputH x inputW]
      static Tensor3D<T> backpropagate(const Tensor3D<T>& dOut, const Tensor3D<T>& input,
                                       const ConvLayerConfig& config, const ConvParameters<T>& params,
                                       std::vector<T>& dFilters, std::vector<T>& dBiases);
  };
}

//===================================================================================================================//

#endif // CNN_CONV2D_HPP

