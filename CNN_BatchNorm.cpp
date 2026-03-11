#include "CNN_BatchNorm.hpp"

#include <cmath>

using namespace CNN;

//===================================================================================================================//

template <typename T>
void BatchNorm<T>::propagate(std::vector<Tensor3D<T>*>& batch, const Shape3D& shape, NormParameters<T>& params,
                             const NormLayerConfig& config, bool training, std::vector<Tensor3D<T>>* xNormalized,
                             std::vector<T>* batchMean, std::vector<T>* batchVar)
{
  ulong C = shape.c;
  ulong H = shape.h;
  ulong W = shape.w;
  ulong spatialSize = H * W;
  ulong N = batch.size();
  T eps = static_cast<T>(config.epsilon);

  if (!training) {
    // Inference: use running mean/var to normalize each sample independently
    for (ulong n = 0; n < N; n++) {
      Tensor3D<T>& sample = *batch[n];

      for (ulong c = 0; c < C; c++) {
        T mean = params.runningMean[c];
        T var = params.runningVar[c];
        T gamma = params.gamma[c];
        T beta = params.beta[c];
        T invStd = static_cast<T>(1) / std::sqrt(var + eps);

        for (ulong s = 0; s < spatialSize; s++) {
          ulong idx = c * spatialSize + s;
          sample.data[idx] = gamma * (sample.data[idx] - mean) * invStd + beta;
        }
      }
    }
  } else {
    // Training: compute batch-wide statistics across all N samples
    T momentum = static_cast<T>(config.momentum);
    T nTotal = static_cast<T>(N * spatialSize);

    batchMean->resize(C);
    batchVar->resize(C);
    xNormalized->resize(N, Tensor3D<T>(shape));

    for (ulong c = 0; c < C; c++) {
      // Compute mean across all N samples and spatial dims
      T mean = static_cast<T>(0);

      for (ulong n = 0; n < N; n++) {
        const Tensor3D<T>& sample = *batch[n];

        for (ulong s = 0; s < spatialSize; s++) {
          mean += sample.data[c * spatialSize + s];
        }
      }

      mean /= nTotal;

      // Compute variance across all N samples and spatial dims
      T var = static_cast<T>(0);

      for (ulong n = 0; n < N; n++) {
        const Tensor3D<T>& sample = *batch[n];

        for (ulong s = 0; s < spatialSize; s++) {
          T diff = sample.data[c * spatialSize + s] - mean;
          var += diff * diff;
        }
      }

      var /= nTotal;

      (*batchMean)[c] = mean;
      (*batchVar)[c] = var;

      // Normalize, scale, and shift all samples
      T invStd = static_cast<T>(1) / std::sqrt(var + eps);
      T gamma = params.gamma[c];
      T beta = params.beta[c];

      for (ulong n = 0; n < N; n++) {
        Tensor3D<T>& sample = *batch[n];

        for (ulong s = 0; s < spatialSize; s++) {
          ulong idx = c * spatialSize + s;
          (*xNormalized)[n].data[idx] = (sample.data[idx] - mean) * invStd;
          sample.data[idx] = gamma * (*xNormalized)[n].data[idx] + beta;
        }
      }

      // Update running statistics
      params.runningMean[c] = (static_cast<T>(1) - momentum) * params.runningMean[c] + momentum * mean;
      params.runningVar[c] = (static_cast<T>(1) - momentum) * params.runningVar[c] + momentum * var;
    }
  }
}

//===================================================================================================================//

template <typename T>
void BatchNorm<T>::backpropagate(std::vector<Tensor3D<T>*>& dOutputs, const Shape3D& shape,
                                 const NormParameters<T>& params, const NormLayerConfig& config,
                                 const std::vector<T>& batchMean, const std::vector<T>& batchVar,
                                 const std::vector<Tensor3D<T>>& xNormalized, std::vector<T>& dGamma,
                                 std::vector<T>& dBeta)
{
  ulong C = shape.c;
  ulong H = shape.h;
  ulong W = shape.w;
  ulong spatialSize = H * W;
  ulong N = dOutputs.size();
  T eps = static_cast<T>(config.epsilon);
  T nTotal = static_cast<T>(N * spatialSize);

  dGamma.resize(C);
  dBeta.resize(C);

  for (ulong c = 0; c < C; c++) {
    T gamma = params.gamma[c];
    T var = batchVar[c];
    T invStd = static_cast<T>(1) / std::sqrt(var + eps);

    // Compute dGamma and dBeta across all samples
    T dg = static_cast<T>(0);
    T db = static_cast<T>(0);

    for (ulong n = 0; n < N; n++) {
      const Tensor3D<T>& dOut = *dOutputs[n];

      for (ulong s = 0; s < spatialSize; s++) {
        ulong idx = c * spatialSize + s;
        dg += dOut.data[idx] * xNormalized[n].data[idx];
        db += dOut.data[idx];
      }
    }

    dGamma[c] = dg;
    dBeta[c] = db;

    // Compute dInput for all samples using the full batch norm gradient formula
    // dx[n,c,s] = (gamma / (N_total * sqrt(var + eps))) * (N_total * dout[n,c,s] - db - xnorm[n,c,s] * dg)
    for (ulong n = 0; n < N; n++) {
      Tensor3D<T>& dOut = *dOutputs[n];

      for (ulong s = 0; s < spatialSize; s++) {
        ulong idx = c * spatialSize + s;
        dOut.data[idx] = (gamma * invStd / nTotal) * (nTotal * dOut.data[idx] - db - xNormalized[n].data[idx] * dg);
      }
    }
  }
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::BatchNorm<int>;
template class CNN::BatchNorm<double>;
template class CNN::BatchNorm<float>;
