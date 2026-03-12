#include "CNN_GlobalAvgPool.hpp"

using namespace CNN;

//===================================================================================================================//

// Forward: average each channel's spatial dimensions to produce a (C, 1, 1) output.
// The input tensor is modified in-place: data is compacted to the first C elements.
template <typename T>
void GlobalAvgPool<T>::propagate(Tensor3D<T>& input, const Shape3D& inputShape)
{
  ulong C = inputShape.c;
  ulong spatialSize = inputShape.h * inputShape.w;

  for (ulong c = 0; c < C; c++) {
    T sum = static_cast<T>(0);
    ulong base = c * spatialSize;

    for (ulong s = 0; s < spatialSize; s++)
      sum += input.data[base + s];

    input.data[c] = sum / static_cast<T>(spatialSize);
  }

  input.data.resize(C);
  input.shape = {C, 1, 1};
}

//===================================================================================================================//

// Backward: distribute the gradient evenly across all spatial positions.
// gradOutput has shape (C, 1, 1) on entry; expanded to (C, H, W) on exit.
template <typename T>
void GlobalAvgPool<T>::backpropagate(Tensor3D<T>& gradOutput, const Shape3D& inputShape)
{
  ulong C = inputShape.c;
  ulong spatialSize = inputShape.h * inputShape.w;
  T invSpatial = static_cast<T>(1) / static_cast<T>(spatialSize);

  // Save the per-channel gradients before resizing
  std::vector<T> channelGrads(C);

  for (ulong c = 0; c < C; c++)
    channelGrads[c] = gradOutput.data[c] * invSpatial;

  gradOutput.data.resize(C * spatialSize);
  gradOutput.shape = inputShape;

  for (ulong c = 0; c < C; c++) {
    T grad = channelGrads[c];
    ulong base = c * spatialSize;

    for (ulong s = 0; s < spatialSize; s++)
      gradOutput.data[base + s] = grad;
  }
}

//===================================================================================================================//

template class GlobalAvgPool<int>;
template class GlobalAvgPool<float>;
template class GlobalAvgPool<double>;
