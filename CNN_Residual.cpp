#include "CNN_Residual.hpp"

using namespace CNN;

//===================================================================================================================//

// Forward: blockOutput += project(skipInput) or blockOutput += skipInput
template <typename T>
void Residual<T>::propagate(Tensor3D<T>& blockOutput, const Tensor3D<T>& skipInput,
                            const ResidualParameters<T>* projection)
{
  if (projection == nullptr) {
    // Identity shortcut: element-wise addition (same channels)
    for (ulong i = 0; i < blockOutput.data.size(); i++)
      blockOutput.data[i] += skipInput.data[i];
  } else {
    // 1×1 projection shortcut: project skip from inC to outC, then add
    ulong inC = projection->inC;
    ulong outC = projection->outC;
    ulong spatialSize = blockOutput.data.size() / outC;

    for (ulong oc = 0; oc < outC; oc++) {
      for (ulong s = 0; s < spatialSize; s++) {
        T projected = projection->biases[oc];

        for (ulong ic = 0; ic < inC; ic++)
          projected += projection->weights[oc * inC + ic] * skipInput.data[ic * spatialSize + s];

        blockOutput.data[oc * spatialSize + s] += projected;
      }
    }
  }
}

//===================================================================================================================//

// Backward: compute dSkip and (optionally) projection weight gradients.
// The gradient through the block path is just dBlockOutput unchanged (it continues backward).
// The gradient through the skip path is: dSkip = dBlockOutput (identity) or dSkip = W^T * dBlockOutput (projection).
template <typename T>
Tensor3D<T> Residual<T>::backpropagate(const Tensor3D<T>& dBlockOutput, const Tensor3D<T>& skipInput,
                                       const ResidualParameters<T>* projection,
                                       ResidualParameters<T>* dProjection)
{
  if (projection == nullptr) {
    // Identity shortcut: gradient passes through unchanged
    return dBlockOutput; // Copy — dSkip == dBlockOutput
  }

  // Projection shortcut
  ulong inC = projection->inC;
  ulong outC = projection->outC;
  ulong spatialSize = skipInput.data.size() / inC;

  // dSkip = W^T * dOut (per spatial element)
  Shape3D skipShape = {inC, skipInput.shape.h, skipInput.shape.w};
  Tensor3D<T> dSkip(skipShape);

  for (ulong ic = 0; ic < inC; ic++) {
    for (ulong s = 0; s < spatialSize; s++) {
      T sum = static_cast<T>(0);

      for (ulong oc = 0; oc < outC; oc++)
        sum += projection->weights[oc * inC + ic] * dBlockOutput.data[oc * spatialSize + s];

      dSkip.data[ic * spatialSize + s] = sum;
    }
  }

  // Accumulate projection weight gradients: dW += dOut * skip^T, dB += dOut
  if (dProjection != nullptr) {
    for (ulong oc = 0; oc < outC; oc++) {
      for (ulong ic = 0; ic < inC; ic++) {
        T dw = static_cast<T>(0);

        for (ulong s = 0; s < spatialSize; s++)
          dw += dBlockOutput.data[oc * spatialSize + s] * skipInput.data[ic * spatialSize + s];

        dProjection->weights[oc * inC + ic] += dw;
      }

      for (ulong s = 0; s < spatialSize; s++)
        dProjection->biases[oc] += dBlockOutput.data[oc * spatialSize + s];
    }
  }

  return dSkip;
}

//===================================================================================================================//

template class Residual<int>;
template class Residual<float>;
template class Residual<double>;

