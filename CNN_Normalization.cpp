#include "CNN_Normalization.hpp"

#include <cmath>

using namespace CNN;

//===================================================================================================================//

template <typename T>
void Normalization<T>::propagate(std::vector<Tensor3D<T>*>& batch, const Shape3D& shape, NormParameters<T>& params,
                                 const NormLayerConfig& config, LayerType normType, bool training,
                                 std::vector<Tensor3D<T>>* xNormalized, std::vector<T>* statsMean,
                                 std::vector<T>* statsVar)
{
  ulong C = shape.c;
  ulong spatialSize = shape.h * shape.w;
  ulong N = batch.size();
  T eps = static_cast<T>(config.epsilon);
  bool batchWide = (normType == LayerType::BATCHNORM);

  if (!training) {
    // Inference path
    for (ulong n = 0; n < N; n++) {
      Tensor3D<T>& sample = *batch[n];

      for (ulong c = 0; c < C; c++) {
        T mean, var;

        if (batchWide) {
          mean = params.runningMean[c];
          var = params.runningVar[c];
        } else {
          mean = static_cast<T>(0);

          for (ulong s = 0; s < spatialSize; s++)
            mean += sample.data[c * spatialSize + s];

          mean /= static_cast<T>(spatialSize);

          var = static_cast<T>(0);

          for (ulong s = 0; s < spatialSize; s++) {
            T diff = sample.data[c * spatialSize + s] - mean;
            var += diff * diff;
          }

          var /= static_cast<T>(spatialSize);
        }

        T gamma = params.gamma[c];
        T beta = params.beta[c];
        T invStd = static_cast<T>(1) / std::sqrt(var + eps);

        for (ulong s = 0; s < spatialSize; s++) {
          ulong idx = c * spatialSize + s;
          sample.data[idx] = gamma * (sample.data[idx] - mean) * invStd + beta;
        }
      }
    }

    return;
  }

  // Training path
  T momentum = static_cast<T>(config.momentum);

  statsMean->resize(N * C);
  statsVar->resize(N * C);
  xNormalized->resize(N, Tensor3D<T>(shape));

  if (batchWide) {
    // BatchNorm: compute stats across all N samples per channel
    T nTotal = static_cast<T>(N * spatialSize);

    for (ulong c = 0; c < C; c++) {
      T mean = static_cast<T>(0);

      for (ulong n = 0; n < N; n++) {
        const Tensor3D<T>& sample = *batch[n];

        for (ulong s = 0; s < spatialSize; s++)
          mean += sample.data[c * spatialSize + s];
      }

      mean /= nTotal;

      T var = static_cast<T>(0);

      for (ulong n = 0; n < N; n++) {
        const Tensor3D<T>& sample = *batch[n];

        for (ulong s = 0; s < spatialSize; s++) {
          T diff = sample.data[c * spatialSize + s] - mean;
          var += diff * diff;
        }
      }

      var /= nTotal;

      // Store same stats for all N samples
      for (ulong n = 0; n < N; n++) {
        (*statsMean)[n * C + c] = mean;
        (*statsVar)[n * C + c] = var;
      }

      // Normalize, scale, shift
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
  } else {
    // InstanceNorm: compute per-sample stats
    T nTotal = static_cast<T>(spatialSize);

    for (ulong c = 0; c < C; c++) {
      T meanAccum = static_cast<T>(0);
      T varAccum = static_cast<T>(0);

      for (ulong n = 0; n < N; n++) {
        Tensor3D<T>& sample = *batch[n];

        T mean = static_cast<T>(0);

        for (ulong s = 0; s < spatialSize; s++)
          mean += sample.data[c * spatialSize + s];

        mean /= nTotal;

        T var = static_cast<T>(0);

        for (ulong s = 0; s < spatialSize; s++) {
          T diff = sample.data[c * spatialSize + s] - mean;
          var += diff * diff;
        }

        var /= nTotal;

        (*statsMean)[n * C + c] = mean;
        (*statsVar)[n * C + c] = var;

        meanAccum += mean;
        varAccum += var;

        T invStd = static_cast<T>(1) / std::sqrt(var + eps);
        T gamma = params.gamma[c];
        T beta = params.beta[c];

        for (ulong s = 0; s < spatialSize; s++) {
          ulong idx = c * spatialSize + s;
          (*xNormalized)[n].data[idx] = (sample.data[idx] - mean) * invStd;
          sample.data[idx] = gamma * (*xNormalized)[n].data[idx] + beta;
        }
      }

      T avgMean = meanAccum / static_cast<T>(N);
      T avgVar = varAccum / static_cast<T>(N);
      params.runningMean[c] = (static_cast<T>(1) - momentum) * params.runningMean[c] + momentum * avgMean;
      params.runningVar[c] = (static_cast<T>(1) - momentum) * params.runningVar[c] + momentum * avgVar;
    }
  }
}

//===================================================================================================================//

template <typename T>
void Normalization<T>::backpropagate(std::vector<Tensor3D<T>*>& dOutputs, const Shape3D& shape,
                                     const NormParameters<T>& params, const NormLayerConfig& config, LayerType normType,
                                     const std::vector<T>& statsMean, const std::vector<T>& statsVar,
                                     const std::vector<Tensor3D<T>>& xNormalized, std::vector<T>& dGamma,
                                     std::vector<T>& dBeta)
{
  ulong C = shape.c;
  ulong spatialSize = shape.h * shape.w;
  ulong N = dOutputs.size();
  T eps = static_cast<T>(config.epsilon);
  bool batchWide = (normType == LayerType::BATCHNORM);

  dGamma.assign(C, static_cast<T>(0));
  dBeta.assign(C, static_cast<T>(0));

  if (batchWide) {
    // BatchNorm: compute dGamma/dBeta across all N samples, then dInput
    T nTotal = static_cast<T>(N * spatialSize);

    for (ulong c = 0; c < C; c++) {
      T gamma = params.gamma[c];
      T var = statsVar[c]; // All N slices have the same value; use index 0*C+c = c
      T invStd = static_cast<T>(1) / std::sqrt(var + eps);

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

      for (ulong n = 0; n < N; n++) {
        Tensor3D<T>& dOut = *dOutputs[n];

        for (ulong s = 0; s < spatialSize; s++) {
          ulong idx = c * spatialSize + s;
          dOut.data[idx] = (gamma * invStd / nTotal) * (nTotal * dOut.data[idx] - db - xNormalized[n].data[idx] * dg);
        }
      }
    }
  } else {
    // InstanceNorm: compute dGamma/dBeta per-sample using per-sample stats
    T nTotal = static_cast<T>(spatialSize);

    for (ulong c = 0; c < C; c++) {
      T gamma = params.gamma[c];

      for (ulong n = 0; n < N; n++) {
        T var = statsVar[n * C + c];
        T invStd = static_cast<T>(1) / std::sqrt(var + eps);

        Tensor3D<T>& dOut = *dOutputs[n];

        T dg = static_cast<T>(0);
        T db = static_cast<T>(0);

        for (ulong s = 0; s < spatialSize; s++) {
          ulong idx = c * spatialSize + s;
          dg += dOut.data[idx] * xNormalized[n].data[idx];
          db += dOut.data[idx];
        }

        dGamma[c] += dg;
        dBeta[c] += db;

        for (ulong s = 0; s < spatialSize; s++) {
          ulong idx = c * spatialSize + s;
          dOut.data[idx] = (gamma * invStd / nTotal) * (nTotal * dOut.data[idx] - db - xNormalized[n].data[idx] * dg);
        }
      }
    }
  }
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::Normalization<int>;
template class CNN::Normalization<double>;
template class CNN::Normalization<float>;
