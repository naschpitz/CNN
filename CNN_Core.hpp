#ifndef CNN_CORE_HPP
#define CNN_CORE_HPP

#include "CNN_Types.hpp"
#include "CNN_CoreConfig.hpp"
#include "CNN_Sample.hpp"
#include "CNN_TrainingProgress.hpp"
#include "CNN_TrainingMetadata.hpp"
#include "CNN_PredictMetadata.hpp"
#include "CNN_TestResult.hpp"

#include <chrono>
#include <memory>

//===================================================================================================================//

namespace CNN {
  template <typename T>
  class Core {
    public:
      //-- Factory --//
      static std::unique_ptr<Core<T>> makeCore(const CoreConfig<T>& config);

      //-- Core interface --//
      virtual Output<T> predict(const Input<T>& input) = 0;
      virtual void train(const Samples<T>& samples) = 0;
      virtual TestResult<T> test(const Samples<T>& samples) = 0;

      //-- Destructor --//
      virtual ~Core() = default;

      //-- Getters --//
      ModeType getModeType() const { return modeType; }
      DeviceType getDeviceType() const { return deviceType; }
      const Shape3D& getInputShape() const { return inputShape; }
      const LayersConfig& getLayersConfig() const { return layersConfig; }
      const TrainingConfig<T>& getTrainingConfig() const { return trainingConfig; }
      const PredictMetadata<T>& getPredictMetadata() const { return predictMetadata; }
      const TrainingMetadata<T>& getTrainingMetadata() const { return trainingMetadata; }
      const Parameters<T>& getParameters() const { return parameters; }
      const LossFunctionConfig<T>& getLossFunctionConfig() const { return coreConfig.lossFunctionConfig; }

      //-- Setters --//
      void setTrainingCallback(TrainingCallback<T> callback) { trainingCallback = callback; }

      //-- Log level --//
      void setLogLevel(LogLevel level) { logLevel = level; }
      LogLevel getLogLevel() const { return logLevel; }

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
      Shape3D inputShape;
      LayersConfig layersConfig;
      TrainingConfig<T> trainingConfig;
      TrainingMetadata<T> trainingMetadata;
      PredictMetadata<T> predictMetadata;
      Parameters<T> parameters;
      ulong progressReports = 1000;
      LogLevel logLevel = LogLevel::ERROR;

      //-- Internal state --//
      TrainingCallback<T> trainingCallback;

    private:
      //-- Timing state --//
      std::chrono::time_point<std::chrono::system_clock> trainingStartTime;
      std::chrono::time_point<std::chrono::system_clock> predictStartTime;
  };
}

//===================================================================================================================//

#endif // CNN_CORE_HPP

