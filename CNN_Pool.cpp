#include "CNN_Pool.hpp"

#include <limits>

using namespace CNN;

//===================================================================================================================//

template <typename T>
Tensor3D<T> Pool<T>::predict(const Tensor3D<T>& input, const PoolLayerConfig& config, std::vector<ulong>& maxIndices)
{
  const ulong inputC = input.shape.c;
  const ulong inputH = input.shape.h;
  const ulong inputW = input.shape.w;

  const ulong pH = config.poolH;
  const ulong pW = config.poolW;
  const ulong sY = config.strideY;
  const ulong sX = config.strideX;

  const ulong outH = (inputH - pH) / sY + 1;
  const ulong outW = (inputW - pW) / sX + 1;

  Tensor3D<T> output({inputC, outH, outW});

  const bool isMax = (config.poolType == PoolTypeEnum::MAX);

  if (isMax) {
    maxIndices.resize(inputC * outH * outW);
  } else {
    maxIndices.clear();
  }

  for (ulong c = 0; c < inputC; c++) {
    for (ulong oh = 0; oh < outH; oh++) {
      for (ulong ow = 0; ow < outW; ow++) {
        if (isMax) {
          T maxVal = std::numeric_limits<T>::lowest();
          ulong maxIdx = 0;

          for (ulong ph = 0; ph < pH; ph++) {
            for (ulong pw = 0; pw < pW; pw++) {
              ulong ih = oh * sY + ph;
              ulong iw = ow * sX + pw;

              T val = input.at(c, ih, iw);

              if (val > maxVal) {
                maxVal = val;
                maxIdx = c * inputH * inputW + ih * inputW + iw;
              }
            }
          }

          output.at(c, oh, ow) = maxVal;
          maxIndices[c * outH * outW + oh * outW + ow] = maxIdx;
        } else {
          // Average pooling
          T sum = static_cast<T>(0);
          T count = static_cast<T>(pH * pW);

          for (ulong ph = 0; ph < pH; ph++) {
            for (ulong pw = 0; pw < pW; pw++) {
              ulong ih = oh * sY + ph;
              ulong iw = ow * sX + pw;
              sum += input.at(c, ih, iw);
            }
          }

          output.at(c, oh, ow) = sum / count;
        }
      }
    }
  }

  return output;
}

//===================================================================================================================//

template <typename T>
Tensor3D<T> Pool<T>::backpropagate(const Tensor3D<T>& dOut, const Shape3D& inputShape, const PoolLayerConfig& config,
                                   const std::vector<ulong>& maxIndices)
{
  const ulong outC = dOut.shape.c;
  const ulong outH = dOut.shape.h;
  const ulong outW = dOut.shape.w;

  const ulong pH = config.poolH;
  const ulong pW = config.poolW;
  const ulong sY = config.strideY;
  const ulong sX = config.strideX;

  Tensor3D<T> dInput(inputShape);

  const bool isMax = (config.poolType == PoolTypeEnum::MAX);

  for (ulong c = 0; c < outC; c++) {
    for (ulong oh = 0; oh < outH; oh++) {
      for (ulong ow = 0; ow < outW; ow++) {
        T dOutVal = dOut.at(c, oh, ow);

        if (isMax) {
          // Max pooling: gradient goes only to the max element
          ulong maxIdx = maxIndices[c * outH * outW + oh * outW + ow];
          dInput.data[maxIdx] += dOutVal;
        } else {
          // Average pooling: gradient is distributed evenly
          T count = static_cast<T>(pH * pW);

          for (ulong ph = 0; ph < pH; ph++) {
            for (ulong pw = 0; pw < pW; pw++) {
              ulong ih = oh * sY + ph;
              ulong iw = ow * sX + pw;
              dInput.at(c, ih, iw) += dOutVal / count;
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
template class CNN::Pool<int>;
template class CNN::Pool<double>;
template class CNN::Pool<float>;
