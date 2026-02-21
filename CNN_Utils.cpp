#include "CNN_Utils.hpp"
#include "CNN_Core.hpp"

#include <ANN_Utils.hpp>

#include <QDebug>
#include <QFile>

#include <stdexcept>

using namespace CNN;

//===================================================================================================================//

template <typename T>
std::unique_ptr<Core<T>> Utils<T>::load(const std::string& configFilePath) {
  QFile file(QString::fromStdString(configFilePath));

  bool result = file.open(QIODevice::ReadOnly);

  if (!result) {
    throw std::runtime_error("Failed to open config file: " + configFilePath);
  }

  QByteArray fileData = file.readAll();
  std::string jsonString = fileData.toStdString();

  CoreConfig<T> coreConfig;

  try {
    nlohmann::json json = nlohmann::json::parse(jsonString);

    Utils::loadCoreConfig(json, coreConfig);
    coreConfig.inputShape = Utils::loadInputShape(json);
    coreConfig.layersConfig = Utils::loadLayersConfig(json);
    coreConfig.trainingConfig = Utils::loadTrainingConfig(json);
    coreConfig.parameters = Utils::loadParameters(json);

  } catch (const nlohmann::json::parse_error& e) {
    qCritical() << "Config file JSON parse error: " << e.what();
  }

  return Core<T>::makeCore(coreConfig);
}

//===================================================================================================================//

template <typename T>
std::string Utils<T>::save(const Core<T>& core) {
  nlohmann::ordered_json json;

  json["device"] = Device::typeToName(core.getDeviceType());
  json["mode"] = Mode::typeToName(core.getModeType());
  json["inputShape"] = getInputShapeJson(core.getInputShape());
  json["cnnLayersConfig"] = getCNNLayersConfigJson(core.getLayersConfig().cnnLayers);
  json["denseLayersConfig"] = getDenseLayersConfigJson(core.getLayersConfig().denseLayers);
  json["trainingConfig"] = getTrainingConfigJson(core.getTrainingConfig());
  json["trainingMetadata"] = getTrainingMetadataJson(core.getTrainingMetadata());
  json["parameters"] = getParametersJson(core.getParameters());

  return json.dump(4);
}

//===================================================================================================================//

template <typename T>
void Utils<T>::save(const Core<T>& core, const std::string& filePath) {
  QFile file(QString::fromStdString(filePath));

  if (!file.open(QIODevice::WriteOnly)) {
    throw std::runtime_error("Failed to open file for writing: " + filePath);
  }

  std::string jsonString = save(core);
  file.write(jsonString.c_str());
  file.close();
}

//===================================================================================================================//

template <typename T>
Samples<T> Utils<T>::loadSamples(const std::string& samplesFilePath, const Shape3D& inputShape) {
  QFile file(QString::fromStdString(samplesFilePath));

  if (!file.open(QIODevice::ReadOnly)) {
    throw std::runtime_error("Failed to open samples file: " + samplesFilePath);
  }

  QByteArray fileData = file.readAll();
  std::string jsonString = fileData.toStdString();

  nlohmann::json json = nlohmann::json::parse(jsonString);
  const nlohmann::json& samplesArray = json.at("samples");

  Samples<T> samples;
  samples.reserve(samplesArray.size());

  for (const auto& sampleJson : samplesArray) {
    Sample<T> sample;

    // Load input as flat array and reshape to 3D
    std::vector<T> flatInput = sampleJson.at("input").get<std::vector<T>>();

    if (flatInput.size() != inputShape.size()) {
      throw std::runtime_error("Sample input size (" + std::to_string(flatInput.size()) +
        ") does not match expected input shape size (" + std::to_string(inputShape.size()) + ")");
    }

    sample.input = Input<T>(inputShape);
    sample.input.data = std::move(flatInput);

    // Load output
    sample.output = sampleJson.at("output").get<Output<T>>();

    samples.push_back(std::move(sample));
  }

  return samples;
}

//===================================================================================================================//

template <typename T>
void Utils<T>::loadCoreConfig(const nlohmann::json& json, CoreConfig<T>& coreConfig) {
  if (json.contains("device")) {
    std::string deviceName = json.at("device").get<std::string>();
    coreConfig.deviceType = Device::nameToType(deviceName);
  } else {
    coreConfig.deviceType = DeviceType::CPU;
  }

  if (json.contains("mode")) {
    std::string modeName = json.at("mode").get<std::string>();
    coreConfig.modeType = Mode::nameToType(modeName);
  } else {
    coreConfig.modeType = ModeType::PREDICT;
  }
}

//===================================================================================================================//

template <typename T>
Shape3D Utils<T>::loadInputShape(const nlohmann::json& json) {
  const nlohmann::json& shapeJson = json.at("inputShape");

  Shape3D shape;
  shape.c = shapeJson.at("c").get<ulong>();
  shape.h = shapeJson.at("h").get<ulong>();
  shape.w = shapeJson.at("w").get<ulong>();

  return shape;
}

//===================================================================================================================//

template <typename T>
LayersConfig Utils<T>::loadLayersConfig(const nlohmann::json& json) {
  LayersConfig config;

  config.cnnLayers = loadCNNLayersConfig(json);
  config.denseLayers = loadDenseLayersConfig(json);

  return config;
}

//===================================================================================================================//

template <typename T>
std::vector<CNNLayerConfig> Utils<T>::loadCNNLayersConfig(const nlohmann::json& json) {
  std::vector<CNNLayerConfig> layers;

  if (!json.contains("cnnLayersConfig")) {
    return layers;
  }

  const nlohmann::json& layersArray = json.at("cnnLayersConfig");

  for (const auto& layerJson : layersArray) {
    std::string type = layerJson.at("type").get<std::string>();

    CNNLayerConfig layerConfig;

    if (type == "conv") {
      layerConfig.type = LayerType::CONV;

      ConvLayerConfig conv;
      conv.numFilters = layerJson.at("numFilters").get<ulong>();
      conv.filterH = layerJson.at("filterH").get<ulong>();
      conv.filterW = layerJson.at("filterW").get<ulong>();
      conv.strideY = layerJson.at("strideY").get<ulong>();
      conv.strideX = layerJson.at("strideX").get<ulong>();

      std::string strategy = layerJson.at("slidingStrategy").get<std::string>();
      conv.slidingStrategy = SlidingStrategy::nameToType(strategy);

      layerConfig.config = conv;
    } else if (type == "relu") {
      layerConfig.type = LayerType::RELU;
      layerConfig.config = ReLULayerConfig{};
    } else if (type == "pool") {
      layerConfig.type = LayerType::POOL;

      PoolLayerConfig pool;
      std::string poolTypeName = layerJson.at("poolType").get<std::string>();
      pool.poolType = PoolType::nameToType(poolTypeName);
      pool.poolH = layerJson.at("poolH").get<ulong>();
      pool.poolW = layerJson.at("poolW").get<ulong>();
      pool.strideY = layerJson.at("strideY").get<ulong>();
      pool.strideX = layerJson.at("strideX").get<ulong>();

      layerConfig.config = pool;
    } else if (type == "flatten") {
      layerConfig.type = LayerType::FLATTEN;
      layerConfig.config = FlattenLayerConfig{};
    } else {
      throw std::runtime_error("Unknown CNN layer type: " + type);
    }

    layers.push_back(layerConfig);
  }

  return layers;
}

//===================================================================================================================//

template <typename T>
std::vector<DenseLayerConfig> Utils<T>::loadDenseLayersConfig(const nlohmann::json& json) {
  std::vector<DenseLayerConfig> layers;

  if (!json.contains("denseLayersConfig")) {
    return layers;
  }

  const nlohmann::json& layersArray = json.at("denseLayersConfig");

  for (const auto& layerJson : layersArray) {
    DenseLayerConfig dense;
    dense.numNeurons = layerJson.at("numNeurons").get<ulong>();

    std::string actvFuncName = layerJson.at("actvFunc").get<std::string>();
    dense.actvFuncType = ANN::ActvFunc::nameToType(actvFuncName);

    layers.push_back(dense);
  }

  return layers;
}



//===================================================================================================================//

template <typename T>
TrainingConfig<T> Utils<T>::loadTrainingConfig(const nlohmann::json& json) {
  TrainingConfig<T> config;

  if (!json.contains("trainingConfig")) {
    return config;
  }

  const nlohmann::json& tcJson = json.at("trainingConfig");

  config.numEpochs = tcJson.at("numEpochs").get<ulong>();
  config.learningRate = tcJson.at("learningRate").get<float>();

  if (tcJson.contains("numThreads")) {
    config.numThreads = tcJson.at("numThreads").get<int>();
  }

  if (tcJson.contains("progressReports")) {
    config.progressReports = tcJson.at("progressReports").get<ulong>();
  }

  return config;
}

//===================================================================================================================//

template <typename T>
Parameters<T> Utils<T>::loadParameters(const nlohmann::json& json) {
  Parameters<T> params;

  if (!json.contains("parameters")) {
    return params;
  }

  const nlohmann::json& paramsJson = json.at("parameters");

  // Load conv parameters
  if (paramsJson.contains("conv")) {
    const nlohmann::json& convArray = paramsJson.at("conv");

    for (const auto& convJson : convArray) {
      ConvParameters<T> cp;
      cp.numFilters = convJson.at("numFilters").get<ulong>();
      cp.inputC = convJson.at("inputC").get<ulong>();
      cp.filterH = convJson.at("filterH").get<ulong>();
      cp.filterW = convJson.at("filterW").get<ulong>();
      cp.filters = convJson.at("filters").get<std::vector<T>>();
      cp.biases = convJson.at("biases").get<std::vector<T>>();
      params.convParams.push_back(std::move(cp));
    }
  }

  // Load dense parameters
  if (paramsJson.contains("dense")) {
    const nlohmann::json& denseJson = paramsJson.at("dense");
    params.denseParams.weights = denseJson.at("weights").get<ANN::Tensor3D<T>>();
    params.denseParams.biases = denseJson.at("biases").get<ANN::Tensor2D<T>>();
  }

  return params;
}

//===================================================================================================================//

template <typename T>
nlohmann::ordered_json Utils<T>::getInputShapeJson(const Shape3D& shape) {
  nlohmann::ordered_json json;
  json["c"] = shape.c;
  json["h"] = shape.h;
  json["w"] = shape.w;
  return json;
}

//===================================================================================================================//

template <typename T>
nlohmann::ordered_json Utils<T>::getCNNLayersConfigJson(const std::vector<CNNLayerConfig>& layers) {
  nlohmann::ordered_json arr = nlohmann::ordered_json::array();

  for (const auto& layer : layers) {
    nlohmann::ordered_json layerJson;

    switch (layer.type) {
      case LayerType::CONV: {
        const auto& conv = std::get<ConvLayerConfig>(layer.config);
        layerJson["type"] = "conv";
        layerJson["numFilters"] = conv.numFilters;
        layerJson["filterH"] = conv.filterH;
        layerJson["filterW"] = conv.filterW;
        layerJson["strideY"] = conv.strideY;
        layerJson["strideX"] = conv.strideX;
        layerJson["slidingStrategy"] = SlidingStrategy::typeToName(conv.slidingStrategy);
        break;
      }
      case LayerType::RELU:
        layerJson["type"] = "relu";
        break;
      case LayerType::POOL: {
        const auto& pool = std::get<PoolLayerConfig>(layer.config);
        layerJson["type"] = "pool";
        layerJson["poolType"] = PoolType::typeToName(pool.poolType);
        layerJson["poolH"] = pool.poolH;
        layerJson["poolW"] = pool.poolW;
        layerJson["strideY"] = pool.strideY;
        layerJson["strideX"] = pool.strideX;
        break;
      }
      case LayerType::FLATTEN:
        layerJson["type"] = "flatten";
        break;
    }

    arr.push_back(layerJson);
  }

  return arr;
}

//===================================================================================================================//

template <typename T>
nlohmann::ordered_json Utils<T>::getDenseLayersConfigJson(const std::vector<DenseLayerConfig>& layers) {
  nlohmann::ordered_json arr = nlohmann::ordered_json::array();

  for (const auto& layer : layers) {
    nlohmann::ordered_json layerJson;
    layerJson["numNeurons"] = layer.numNeurons;
    layerJson["actvFunc"] = ANN::ActvFunc::typeToName(layer.actvFuncType);
    arr.push_back(layerJson);
  }

  return arr;
}

//===================================================================================================================//

template <typename T>
nlohmann::ordered_json Utils<T>::getTrainingConfigJson(const TrainingConfig<T>& config) {
  nlohmann::ordered_json json;
  json["numEpochs"] = config.numEpochs;
  json["learningRate"] = config.learningRate;
  return json;
}

//===================================================================================================================//

template <typename T>
nlohmann::ordered_json Utils<T>::getTrainingMetadataJson(const TrainingMetadata<T>& metadata) {
  nlohmann::ordered_json json;
  json["startTime"] = metadata.startTime;
  json["endTime"] = metadata.endTime;
  json["durationSeconds"] = metadata.durationSeconds;
  json["durationFormatted"] = metadata.durationFormatted;
  json["numSamples"] = metadata.numSamples;
  json["finalLoss"] = metadata.finalLoss;
  return json;
}

//===================================================================================================================//

template <typename T>
nlohmann::ordered_json Utils<T>::getParametersJson(const Parameters<T>& params) {
  nlohmann::ordered_json json;

  // Conv parameters
  nlohmann::ordered_json convArr = nlohmann::ordered_json::array();

  for (const auto& cp : params.convParams) {
    nlohmann::ordered_json cpJson;
    cpJson["numFilters"] = cp.numFilters;
    cpJson["inputC"] = cp.inputC;
    cpJson["filterH"] = cp.filterH;
    cpJson["filterW"] = cp.filterW;
    cpJson["filters"] = cp.filters;
    cpJson["biases"] = cp.biases;
    convArr.push_back(cpJson);
  }

  json["conv"] = convArr;

  // Dense parameters
  nlohmann::ordered_json denseJson;
  denseJson["weights"] = params.denseParams.weights;
  denseJson["biases"] = params.denseParams.biases;
  json["dense"] = denseJson;

  return json;
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::Utils<int>;
template class CNN::Utils<double>;
template class CNN::Utils<float>;