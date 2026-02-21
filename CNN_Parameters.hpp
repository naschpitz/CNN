#ifndef CNN_PARAMETERS_HPP
#define CNN_PARAMETERS_HPP

#include "CNN_Types.hpp"

#include <ANN_Parameters.hpp>

#include <vector>

//===================================================================================================================//

namespace CNN {
  // Parameters for a single convolution layer:
  // filters: [numFilters][inputChannels][filterH][filterW] stored as flat vectors
  // biases: [numFilters]
  template <typename T>
  struct ConvParameters {
    std::vector<T> filters;   // Flat: numFilters * inputC * filterH * filterW
    std::vector<T> biases;    // numFilters

    ulong numFilters = 0;
    ulong inputC = 0;
    ulong filterH = 0;
    ulong filterW = 0;

    // Access filter weight: filters[f][c][h][w]
    T& filterAt(ulong f, ulong c, ulong h, ulong w) {
      return filters[f * inputC * filterH * filterW + c * filterH * filterW + h * filterW + w];
    }

    const T& filterAt(ulong f, ulong c, ulong h, ulong w) const {
      return filters[f * inputC * filterH * filterW + c * filterH * filterW + h * filterW + w];
    }
  };

  // All CNN parameters (conv layers + ANN dense parameters)
  template <typename T>
  struct Parameters {
    std::vector<ConvParameters<T>> convParams;  // One per conv layer
    ANN::Parameters<T> denseParams;              // Dense layer parameters (delegated to ANN)
  };
}

//===================================================================================================================//

#endif // CNN_PARAMETERS_HPP

