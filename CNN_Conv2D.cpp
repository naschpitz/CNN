#include "CNN_Conv2D.hpp"
#include "CNN_SlidingStrategy.hpp"

using namespace CNN;

//===================================================================================================================//

template <typename T>
Tensor3D<T> Conv2D<T>::predict(const Tensor3D<T>& input, const ConvLayerConfig& config, const ConvParameters<T>& params)
{
  const ulong inputC = input.shape.c;
  const ulong inputH = input.shape.h;
  const ulong inputW = input.shape.w;

  const ulong kH = config.filterH;
  const ulong kW = config.filterW;
  const ulong sY = config.strideY;
  const ulong sX = config.strideX;
  const ulong numFilters = config.numFilters;

  const ulong padY = SlidingStrategy::computePadding(kH, config.slidingStrategy);
  const ulong padX = SlidingStrategy::computePadding(kW, config.slidingStrategy);

  const ulong outH = (inputH + 2 * padY - kH) / sY + 1;
  const ulong outW = (inputW + 2 * padX - kW) / sX + 1;

  Tensor3D<T> output({numFilters, outH, outW});

  for (ulong f = 0; f < numFilters; f++) {
    for (ulong oh = 0; oh < outH; oh++) {
      for (ulong ow = 0; ow < outW; ow++) {
        T sum = params.biases[f];

        for (ulong c = 0; c < inputC; c++) {
          for (ulong kh = 0; kh < kH; kh++) {
            for (ulong kw = 0; kw < kW; kw++) {
              // Compute input position with padding offset
              long ih = static_cast<long>(oh * sY + kh) - static_cast<long>(padY);
              long iw = static_cast<long>(ow * sX + kw) - static_cast<long>(padX);

              // Skip if outside input bounds (zero-padding)
              if (ih < 0 || ih >= static_cast<long>(inputH) || iw < 0 || iw >= static_cast<long>(inputW)) {
                continue;
              }

              sum += input.at(c, static_cast<ulong>(ih), static_cast<ulong>(iw)) * params.filterAt(f, c, kh, kw);
            }
          }
        }

        output.at(f, oh, ow) = sum;
      }
    }
  }

  return output;
}

//===================================================================================================================//

template <typename T>
Tensor3D<T> Conv2D<T>::backpropagate(const Tensor3D<T>& dOut, const Tensor3D<T>& input, const ConvLayerConfig& config,
                                     const ConvParameters<T>& params, std::vector<T>& dFilters, std::vector<T>& dBiases)
{
  const ulong inputC = input.shape.c;
  const ulong inputH = input.shape.h;
  const ulong inputW = input.shape.w;

  const ulong kH = config.filterH;
  const ulong kW = config.filterW;
  const ulong sY = config.strideY;
  const ulong sX = config.strideX;
  const ulong numFilters = config.numFilters;

  const ulong padY = SlidingStrategy::computePadding(kH, config.slidingStrategy);
  const ulong padX = SlidingStrategy::computePadding(kW, config.slidingStrategy);

  const ulong outH = dOut.shape.h;
  const ulong outW = dOut.shape.w;

  // Initialize gradient accumulators
  dFilters.assign(params.filters.size(), static_cast<T>(0));
  dBiases.assign(params.biases.size(), static_cast<T>(0));

  // Gradient w.r.t. input
  Tensor3D<T> dInput(input.shape);

  for (ulong f = 0; f < numFilters; f++) {
    for (ulong oh = 0; oh < outH; oh++) {
      for (ulong ow = 0; ow < outW; ow++) {
        T dOutVal = dOut.at(f, oh, ow);

        // dBias: sum of dOut over spatial dimensions
        dBiases[f] += dOutVal;

        for (ulong c = 0; c < inputC; c++) {
          for (ulong kh = 0; kh < kH; kh++) {
            for (ulong kw = 0; kw < kW; kw++) {
              long ih = static_cast<long>(oh * sY + kh) - static_cast<long>(padY);
              long iw = static_cast<long>(ow * sX + kw) - static_cast<long>(padX);

              if (ih < 0 || ih >= static_cast<long>(inputH) || iw < 0 || iw >= static_cast<long>(inputW)) {
                continue;
              }

              ulong uih = static_cast<ulong>(ih);
              ulong uiw = static_cast<ulong>(iw);

              // dFilters[f][c][kh][kw] += dOut[f][oh][ow] * input[c][ih][iw]
              ulong filterIdx = f * inputC * kH * kW + c * kH * kW + kh * kW + kw;
              dFilters[filterIdx] += dOutVal * input.at(c, uih, uiw);

              // dInput[c][ih][iw] += dOut[f][oh][ow] * filter[f][c][kh][kw]
              dInput.at(c, uih, uiw) += dOutVal * params.filterAt(f, c, kh, kw);
            }
          }
        }
      }
    }
  }

  return dInput;
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::Conv2D<int>;
template class CNN::Conv2D<double>;
template class CNN::Conv2D<float>;
