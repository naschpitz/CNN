#ifndef CNN_CORECPU_HPP
#define CNN_CORECPU_HPP

#include "CNN_Core.hpp"
#include "CNN_CoreCPUWorker.hpp"

#include <memory>
#include <vector>

//===================================================================================================================//

namespace CNN
{
  template <typename T>
  class CoreCPU : public Core<T>
  {
    public:
      //-- Constructor --//
      CoreCPU(const CoreConfig<T>& config);

      //-- Core interface --//
      Output<T> predict(const Input<T>& input) override;
      void train(ulong numSamples, const SampleProvider<T>& sampleProvider) override;
      TestResult<T> test(ulong numSamples, const SampleProvider<T>& sampleProvider) override;

    private:
      //-- Step-by-step worker (for predict / single-threaded path) --//
      std::unique_ptr<CoreCPUWorker<T>> stepWorker;

      //-- Global CNN gradient accumulators (for merging worker results) --//
      std::vector<std::vector<T>> accumDConvFilters;
      std::vector<std::vector<T>> accumDConvBiases;
      std::vector<std::vector<T>> accumDBNGamma;
      std::vector<std::vector<T>> accumDBNBeta;
      std::vector<std::vector<T>> accumNormMean;
      std::vector<std::vector<T>> accumNormVar;

      //-- Adam optimizer state for CNN conv parameters --//
      std::vector<std::vector<T>> adam_m_filters;
      std::vector<std::vector<T>> adam_v_filters;
      std::vector<std::vector<T>> adam_m_biases;
      std::vector<std::vector<T>> adam_v_biases;
      std::vector<std::vector<T>> adam_m_bn_gamma;
      std::vector<std::vector<T>> adam_v_bn_gamma;
      std::vector<std::vector<T>> adam_m_bn_beta;
      std::vector<std::vector<T>> adam_v_bn_beta;
      ulong adam_t = 0;

      //-- Training helpers --//
      void resetGlobalCNNAccumulators();
      void mergeWorkerCNNAccumulators(const CoreCPUWorker<T>& worker);
      void updateCNNParameters(ulong numSamples);
      void updateNormRunningStats(ulong numSamples);
      void allocateAdamState();
  };
}

//===================================================================================================================//

#endif // CNN_CORECPU_HPP
