#include "CNN_InstanceNorm.hpp"

#include <cmath>

using namespace CNN;

//===================================================================================================================//

template <typename T>
Tensor3D<T> InstanceNorm<T>::propagate(const Tensor3D<T>& input, const Shape3D& inputShape, NormParameters<T>& params,
                                       const NormLayerConfig& config, std::vector<T>* batchMean,
                                       std::vector<T>* batchVar, Tensor3D<T>* xNormalized)
{
  ulong C = inputShape.c;
  ulong H = inputShape.h;
  ulong W = inputShape.w;
  ulong spatialSize = H * W;
  T eps = static_cast<T>(config.epsilon);
  bool storeIntermediates = (batchMean != nullptr && batchVar != nullptr && xNormalized != nullptr);

  Tensor3D<T> output(inputShape);

  if (storeIntermediates) {
    batchMean->resize(C);
    batchVar->resize(C);
    *xNormalized = Tensor3D<T>(inputShape);
  }

  for (ulong c = 0; c < C; c++) {
    // Compute per-image spatial mean
    T mean = static_cast<T>(0);

    for (ulong s = 0; s < spatialSize; s++) {
      mean += input.data[c * spatialSize + s];
    }

    mean /= static_cast<T>(spatialSize);

    // Compute per-image spatial variance
    T var = static_cast<T>(0);

    for (ulong s = 0; s < spatialSize; s++) {
      T diff = input.data[c * spatialSize + s] - mean;
      var += diff * diff;
    }

    var /= static_cast<T>(spatialSize);

    // Normalize, scale, and shift
    T invStd = static_cast<T>(1) / std::sqrt(var + eps);
    T gamma = params.gamma[c];
    T beta = params.beta[c];

    if (storeIntermediates) {
      (*batchMean)[c] = mean;
      (*batchVar)[c] = var;

      for (ulong s = 0; s < spatialSize; s++) {
        ulong idx = c * spatialSize + s;
        xNormalized->data[idx] = (input.data[idx] - mean) * invStd;
        output.data[idx] = gamma * xNormalized->data[idx] + beta;
      }

      // Update running statistics
      T momentum = static_cast<T>(config.momentum);
      params.runningMean[c] = (static_cast<T>(1) - momentum) * params.runningMean[c] + momentum * mean;
      params.runningVar[c] = (static_cast<T>(1) - momentum) * params.runningVar[c] + momentum * var;
    } else {
      for (ulong s = 0; s < spatialSize; s++) {
        ulong idx = c * spatialSize + s;
        output.data[idx] = gamma * (input.data[idx] - mean) * invStd + beta;
      }
    }
  }

  return output;
}

//===================================================================================================================//

template <typename T>
Tensor3D<T> InstanceNorm<T>::backpropagate(const Tensor3D<T>& dOutput, const Shape3D& inputShape,
                                           const NormParameters<T>& params, const NormLayerConfig& config,
                                           const std::vector<T>& batchMean, const std::vector<T>& batchVar,
                                           const Tensor3D<T>& xNormalized, std::vector<T>& dGamma,
                                           std::vector<T>& dBeta)
{
  ulong C = inputShape.c;
  ulong H = inputShape.h;
  ulong W = inputShape.w;
  ulong spatialSize = H * W;
  T eps = static_cast<T>(config.epsilon);

  dGamma.resize(C);
  dBeta.resize(C);

  Tensor3D<T> dInput(inputShape);

  for (ulong c = 0; c < C; c++) {
    T gamma = params.gamma[c];
    T var = batchVar[c];
    T invStd = static_cast<T>(1) / std::sqrt(var + eps);
    T N = static_cast<T>(spatialSize);

    // Compute dGamma and dBeta
    T dg = static_cast<T>(0);
    T db = static_cast<T>(0);

    for (ulong s = 0; s < spatialSize; s++) {
      ulong idx = c * spatialSize + s;
      dg += dOutput.data[idx] * xNormalized.data[idx];
      db += dOutput.data[idx];
    }

    dGamma[c] = dg;
    dBeta[c] = db;

    // Compute dInput using the full instance norm gradient formula
    for (ulong s = 0; s < spatialSize; s++) {
      ulong idx = c * spatialSize + s;
      dInput.data[idx] = (gamma * invStd / N) * (N * dOutput.data[idx] - db - xNormalized.data[idx] * dg);
    }
  }

  return dInput;
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::InstanceNorm<int>;
template class CNN::InstanceNorm<double>;
template class CNN::InstanceNorm<float>;
