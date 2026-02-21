#ifndef CNN_UTILS_HPP
#define CNN_UTILS_HPP

#include <json.hpp>
#include "CNN_Core.hpp"

#include <memory>
#include <string>

//===================================================================================================================//

namespace CNN {
  template <typename T>
  class Utils
  {
    public:
      static std::unique_ptr<Core<T>> load(const std::string& configFilePath);

      // Save model with training metadata
      static std::string save(const Core<T>& core);
      static void save(const Core<T>& core, const std::string& filePath);

      // Load samples from JSON file
      static Samples<T> loadSamples(const std::string& samplesFilePath, const Shape3D& inputShape);

    private:
      static void loadCoreConfig(const nlohmann::json& json, CoreConfig<T>& coreConfig);
      static Shape3D loadInputShape(const nlohmann::json& json);
      static LayersConfig loadLayersConfig(const nlohmann::json& json);
      static std::vector<CNNLayerConfig> loadCNNLayersConfig(const nlohmann::json& json);
      static std::vector<DenseLayerConfig> loadDenseLayersConfig(const nlohmann::json& json);
      static TrainingConfig<T> loadTrainingConfig(const nlohmann::json& json);
      static Parameters<T> loadParameters(const nlohmann::json& json);

      static nlohmann::ordered_json getInputShapeJson(const Shape3D& shape);
      static nlohmann::ordered_json getCNNLayersConfigJson(const std::vector<CNNLayerConfig>& layers);
      static nlohmann::ordered_json getDenseLayersConfigJson(const std::vector<DenseLayerConfig>& layers);
      static nlohmann::ordered_json getTrainingConfigJson(const TrainingConfig<T>& config);
      static nlohmann::ordered_json getTrainingMetadataJson(const TrainingMetadata<T>& metadata);
      static nlohmann::ordered_json getParametersJson(const Parameters<T>& params);
  };
}

//===================================================================================================================//

#endif // CNN_UTILS_HPP

