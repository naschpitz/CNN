#include "CNN_GlobalDualPool.hpp"

#include <limits>

using namespace CNN;

//===================================================================================================================//

// Forward: compute avg and max for each channel, output (2C, 1, 1).
// First C elements = channel averages, next C elements = channel maxes.
template <typename T>
void GlobalDualPool<T>::propagate(Tensor3D<T>& input, const Shape3D& inputShape)
{
  ulong C = inputShape.c;
  ulong spatialSize = inputShape.h * inputShape.w;

  std::vector<T> output(2 * C);

  for (ulong c = 0; c < C; c++) {
    T sum = static_cast<T>(0);
    T maxVal = -std::numeric_limits<T>::max();
    ulong base = c * spatialSize;

    for (ulong s = 0; s < spatialSize; s++) {
      T val = input.data[base + s];
      sum += val;

      if (val > maxVal)
        maxVal = val;
    }

    output[c] = sum / static_cast<T>(spatialSize);
    output[C + c] = maxVal;
  }

  input.data = std::move(output);
  input.shape = {2 * C, 1, 1};
}

//===================================================================================================================//

// Backward: distribute gradient from (2C, 1, 1) back to (C, H, W).
// Avg path: gradient distributed uniformly (1/spatialSize).
// Max path: gradient flows only to the first max element.
template <typename T>
void GlobalDualPool<T>::backpropagate(Tensor3D<T>& gradOutput, const Tensor3D<T>& layerInput, const Shape3D& inputShape)
{
  ulong C = inputShape.c;
  ulong spatialSize = inputShape.h * inputShape.w;
  T invSpatial = static_cast<T>(1) / static_cast<T>(spatialSize);

  // Save per-channel gradients before resizing
  std::vector<T> dAvg(C);
  std::vector<T> dMax(C);

  for (ulong c = 0; c < C; c++) {
    dAvg[c] = gradOutput.data[c];
    dMax[c] = gradOutput.data[C + c];
  }

  gradOutput.data.resize(C * spatialSize);
  gradOutput.shape = inputShape;

  for (ulong c = 0; c < C; c++) {
    ulong base = c * spatialSize;

    // Find max index in the original input
    T maxVal = -std::numeric_limits<T>::max();
    ulong maxIdx = 0;

    for (ulong s = 0; s < spatialSize; s++) {
      T val = layerInput.data[base + s];

      if (val > maxVal) {
        maxVal = val;
        maxIdx = s;
      }
    }

    // Avg gradient: uniform distribution
    T avgGrad = dAvg[c] * invSpatial;

    for (ulong s = 0; s < spatialSize; s++)
      gradOutput.data[base + s] = avgGrad;

    // Max gradient: add to the max element only
    gradOutput.data[base + maxIdx] += dMax[c];
  }
}

//===================================================================================================================//

template class GlobalDualPool<int>;
template class GlobalDualPool<float>;
template class GlobalDualPool<double>;
