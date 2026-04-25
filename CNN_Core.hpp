#ifndef CNN_CORE_HPP
#define CNN_CORE_HPP

#include "CNN_Types.hpp"
#include "CNN_CoreConfig.hpp"
#include "CNN_Sample.hpp"
#include "CNN_TrainingProgress.hpp"
#include "CNN_TrainingMetadata.hpp"
#include "CNN_PredictMetadata.hpp"
#include "CNN_PredictResult.hpp"
#include "CNN_ProgressCallback.hpp"
#include "CNN_TestResult.hpp"

#include <atomic>
#include <chrono>
#include <memory>

//===================================================================================================================//

namespace CNN
{
  template <typename T>
  class Core
  {
    public:
      //-- Factory --//
      static std::unique_ptr<Core<T>> makeCore(const CoreConfig<T>& config);

      //-- Core interface --//
      // Streaming predict: pulls inputs in batches from `provider`. Lets
      // callers score arbitrarily large datasets without holding the full
      // Inputs<T> in memory. Eager overloads below wrap this one.
      virtual PredictResults<T> predict(ulong numSamples, const InputProvider<T>& provider) = 0;
      // Eager overloads (concrete wrappers around the streaming version).
      // Each result carries .output (post-activation, e.g. softmax probs)
      // and .logits (pre-activation z of the dense head's last layer, used
      // for calibration / OOD scores that softmax discards).
      PredictResults<T> predict(const Inputs<T>& inputs);
      PredictResult<T> predict(const Input<T>& input);
      virtual void train(ulong numSamples, const SampleProvider<T>& sampleProvider) = 0;
      virtual TestResult<T> test(ulong numSamples, const SampleProvider<T>& sampleProvider) = 0;

      //-- Destructor --//
      virtual ~Core() = default;

      //-- Getters --//
      ModeType getModeType() const
      {
        return modeType;
      }

      DeviceType getDeviceType() const
      {
        return deviceType;
      }

      int getNumThreads() const
      {
        return numThreads;
      }

      int getNumGPUs() const
      {
        return numGPUs;
      }

      const Shape3D& getInputShape() const
      {
        return inputShape;
      }

      const LayersConfig& getLayersConfig() const
      {
        return layersConfig;
      }

      const TrainingConfig<T>& getTrainingConfig() const
      {
        return trainingConfig;
      }

      const PredictMetadata<T>& getPredictMetadata() const
      {
        return predictMetadata;
      }

      const TrainingMetadata<T>& getTrainingMetadata() const
      {
        return trainingMetadata;
      }

      const Parameters<T>& getParameters() const
      {
        return parameters;
      }

      const CostFunctionConfig<T>& getCostFunctionConfig() const
      {
        return coreConfig.costFunctionConfig;
      }

      //-- Setters --//
      void setParameters(const Parameters<T>& params)
      {
        parameters = params;
      }

      // Upload current host-side parameters to GPU buffers (no-op for CPU cores).
      virtual void syncParametersToGPU() {}

      void setTrainingCallback(TrainingCallback<T> callback)
      {
        trainingCallback = callback;
      }

      void setProgressCallback(ProgressCallback callback)
      {
        progressCallback = callback;
      }

      // Request early training termination. The training loop checks this at epoch boundaries.
      void requestStop()
      {
        stopRequested.store(true);
      }

      bool isStopRequested() const
      {
        return stopRequested.load();
      }

      //-- Log level --//
      void setLogLevel(LogLevel level)
      {
        logLevel = level;
      }

      LogLevel getLogLevel() const
      {
        return logLevel;
      }

    protected:
      //-- Constructor / Validation --//
      explicit Core(const CoreConfig<T>& coreConfig);
      void sanityCheck(const CoreConfig<T>& coreConfig);

      //-- Training timing --//
      void trainingStart(ulong numSamples);
      TrainingMetadata<T> trainingEnd();

      //-- Predict timing --//
      void predictStart();
      PredictMetadata<T> predictEnd();

      //-- Configuration members --//
      CoreConfig<T> coreConfig;
      DeviceType deviceType;
      ModeType modeType;
      int numThreads = 0;
      int numGPUs = 0;
      Shape3D inputShape;
      LayersConfig layersConfig;
      TrainingConfig<T> trainingConfig;
      TestConfig testConfig;
      TrainingMetadata<T> trainingMetadata;
      PredictMetadata<T> predictMetadata;
      Parameters<T> parameters;
      ulong progressReports = 1000;
      LogLevel logLevel = LogLevel::ERROR;

      //-- Internal state --//
      TrainingCallback<T> trainingCallback;
      ProgressCallback progressCallback;
      std::atomic<bool> stopRequested{false};

    private:
      //-- Timing state --//
      std::chrono::time_point<std::chrono::system_clock> trainingStartTime;
      std::chrono::time_point<std::chrono::system_clock> predictStartTime;
  };
}

//===================================================================================================================//

#endif // CNN_CORE_HPP
