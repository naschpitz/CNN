#include "CNN_BatchNorm.hpp"

#include <cmath>

using namespace CNN;

//===================================================================================================================//

template <typename T>
std::vector<Tensor3D<T>> BatchNorm<T>::propagate(const std::vector<Tensor3D<T>>& inputs, const Shape3D& inputShape,
                                                 NormParameters<T>& params, const NormLayerConfig& config,
                                                 std::vector<T>* batchMean, std::vector<T>* batchVar,
                                                 std::vector<Tensor3D<T>>* xNormalized)
{
  ulong N = inputs.size();
  ulong C = inputShape.c;
  ulong spatialSize = inputShape.h * inputShape.w;
  T eps = static_cast<T>(config.epsilon);
  bool training = (batchMean != nullptr && batchVar != nullptr && xNormalized != nullptr);

  std::vector<Tensor3D<T>> outputs(N, Tensor3D<T>(inputShape));

  if (training) {
    // Training: compute batch-wide mean/var across N×H×W per channel
    ulong M = N * spatialSize;

    std::vector<T> mean(C, static_cast<T>(0));
    std::vector<T> var(C, static_cast<T>(0));

    xNormalized->resize(N, Tensor3D<T>(inputShape));

    // Compute batch mean per channel
    for (ulong c = 0; c < C; c++) {
      T sum = static_cast<T>(0);

      for (ulong n = 0; n < N; n++)

        for (ulong s = 0; s < spatialSize; s++)
          sum += inputs[n].data[c * spatialSize + s];

      mean[c] = sum / static_cast<T>(M);
    }

    // Compute batch variance per channel
    for (ulong c = 0; c < C; c++) {
      T sumSq = static_cast<T>(0);

      for (ulong n = 0; n < N; n++) {
        for (ulong s = 0; s < spatialSize; s++) {
          T diff = inputs[n].data[c * spatialSize + s] - mean[c];
          sumSq += diff * diff;
        }
      }

      var[c] = sumSq / static_cast<T>(M);
    }

    // Normalize, scale, and shift all samples
    for (ulong c = 0; c < C; c++) {
      T invStd = static_cast<T>(1) / std::sqrt(var[c] + eps);
      T gamma = params.gamma[c];
      T beta = params.beta[c];

      for (ulong n = 0; n < N; n++) {
        for (ulong s = 0; s < spatialSize; s++) {
          ulong idx = c * spatialSize + s;
          T xn = (inputs[n].data[idx] - mean[c]) * invStd;
          (*xNormalized)[n].data[idx] = xn;
          outputs[n].data[idx] = gamma * xn + beta;
        }
      }
    }

    // Update running statistics
    T momentum = static_cast<T>(config.momentum);

    for (ulong c = 0; c < C; c++) {
      params.runningMean[c] = (static_cast<T>(1) - momentum) * params.runningMean[c] + momentum * mean[c];
      params.runningVar[c] = (static_cast<T>(1) - momentum) * params.runningVar[c] + momentum * var[c];
    }

    *batchMean = std::move(mean);
    *batchVar = std::move(var);
  } else {
    // Inference: use running stats
    for (ulong c = 0; c < C; c++) {
      T mean = params.runningMean[c];
      T var = params.runningVar[c];
      T invStd = static_cast<T>(1) / std::sqrt(var + eps);
      T gamma = params.gamma[c];
      T beta = params.beta[c];

      for (ulong n = 0; n < N; n++) {
        for (ulong s = 0; s < spatialSize; s++) {
          ulong idx = c * spatialSize + s;
          outputs[n].data[idx] = gamma * (inputs[n].data[idx] - mean) * invStd + beta;
        }
      }
    }
  }

  return outputs;
}

//===================================================================================================================//

template <typename T>
std::vector<Tensor3D<T>>
BatchNorm<T>::backpropagate(const std::vector<Tensor3D<T>>& dOutputs, const Shape3D& inputShape,
                            const NormParameters<T>& params, const NormLayerConfig& config,
                            const std::vector<T>& batchMean, const std::vector<T>& batchVar,
                            const std::vector<Tensor3D<T>>& xNormalized, std::vector<T>& dGamma, std::vector<T>& dBeta)
{
  ulong N = dOutputs.size();
  ulong C = inputShape.c;
  ulong spatialSize = inputShape.h * inputShape.w;
  T M = static_cast<T>(N * spatialSize); // Total elements per channel across all samples
  T eps = static_cast<T>(config.epsilon);

  dGamma.resize(C);
  dBeta.resize(C);

  std::vector<Tensor3D<T>> dInputs(N, Tensor3D<T>(inputShape));

  for (ulong c = 0; c < C; c++) {
    T gamma = params.gamma[c];
    T var = batchVar[c];
    T invStd = static_cast<T>(1) / std::sqrt(var + eps);

    // Compute dGamma and dBeta across all samples
    T dg = static_cast<T>(0);
    T db = static_cast<T>(0);

    for (ulong n = 0; n < N; n++) {
      for (ulong s = 0; s < spatialSize; s++) {
        ulong idx = c * spatialSize + s;
        dg += dOutputs[n].data[idx] * xNormalized[n].data[idx];
        db += dOutputs[n].data[idx];
      }
    }

    dGamma[c] = dg;
    dBeta[c] = db;

    // Compute dInput for all samples using the batch norm gradient formula:
    // dInput[n][idx] = (gamma * invStd / M) * (M * dOut[n][idx] - dBeta - xNorm[n][idx] * dGamma)
    for (ulong n = 0; n < N; n++) {
      for (ulong s = 0; s < spatialSize; s++) {
        ulong idx = c * spatialSize + s;
        dInputs[n].data[idx] = (gamma * invStd / M) * (M * dOutputs[n].data[idx] - db - xNormalized[n].data[idx] * dg);
      }
    }
  }

  return dInputs;
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::BatchNorm<int>;
template class CNN::BatchNorm<double>;
template class CNN::BatchNorm<float>;
