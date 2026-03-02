#include "CNN_GPUKernelBuilder.hpp"
#include "CNN_SlidingStrategy.hpp"

#include <cmath>
#include <string>

using namespace CNN;

//===================================================================================================================//

template <typename T>
GPUKernelBuilder<T>::GPUKernelBuilder(OpenCLWrapper::Core* core, const CoreConfig<T>& coreConfig,
                                      GPUBufferManager<T>& bufferManager, LogLevel logLevel)
  : core(core),
    coreConfig(coreConfig),
    bufferManager(bufferManager),
    logLevel(logLevel)
{
}

//===================================================================================================================//
//-- Kernel setup --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::setupPredictKernels()
{
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  this->addPropagateKernels();
  this->addCopyBridgeKernels();
  this->bufferManager.annGPUWorker->kernelBuilder->addPropagateKernels();

  this->predictKernelsSetup = true;
}

//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::setupTrainingKernels()
{
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  // Propagate pipeline: CNN propagate → copy → ANN propagate
  this->addPropagateKernels();
  this->addCopyBridgeKernels();
  this->bufferManager.annGPUWorker->kernelBuilder->addPropagateKernels();

  // Backpropagate pipeline: ANN backpropagate (with input gradients) → reverse bridge → CNN backpropagate
  this->bufferManager.annGPUWorker->kernelBuilder->addBackpropagateKernels(true);

  // Reverse bridge: copy ANN input gradients to CNN gradient buffer
  ulong lastLayerIdx = this->bufferManager.layerInfos.size() - 1;
  ulong cnnOutputOffset = this->bufferManager.layerInfos[lastLayerIdx].actvOffset;
  this->core->addKernel("copy_ann_grad_to_cnn", this->bufferManager.flattenSize, 0);
  this->core->template addArgument<T>("copy_ann_grad_to_cnn", "dCost_dActvs");
  this->core->template addArgument<T>("copy_ann_grad_to_cnn", "cnn_grads");
  this->core->template addArgument<ulong>("copy_ann_grad_to_cnn", cnnOutputOffset);
  this->core->template addArgument<ulong>("copy_ann_grad_to_cnn", this->bufferManager.flattenSize);

  this->addBackpropagateKernels();

  // Accumulate: CNN + ANN
  this->addCNNAccumulateKernels();
  this->bufferManager.annGPUWorker->kernelBuilder->addAccumulateKernels();

  // Loss: compute weighted MSE on GPU and accumulate into accum_loss buffer
  ulong outputActvOffset = this->bufferManager.annGPUWorker->bufferManager->getOutputActvOffset();
  ulong numOutputNeurons = this->bufferManager.annGPUWorker->bufferManager->getNumOutputNeurons();
  this->core->addKernel("calculate_sample_loss", "calculate_sample_loss", 1, 0);
  this->core->template addArgument<T>("calculate_sample_loss", "actvs");
  this->core->template addArgument<T>("calculate_sample_loss", "outputs");
  this->core->template addArgument<T>("calculate_sample_loss", "lossWeights");
  this->core->template addArgument<T>("calculate_sample_loss", "accum_loss");
  this->core->template addArgument<ulong>("calculate_sample_loss", outputActvOffset);
  this->core->template addArgument<ulong>("calculate_sample_loss", numOutputNeurons);
  this->core->template addArgument<ulong>("calculate_sample_loss",
                                          static_cast<ulong>(this->coreConfig.costFunctionConfig.type));

  this->trainingKernelsSetup = true;
}

//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::setupUpdateKernels(ulong numSamples)
{
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  this->addCNNUpdateKernels(numSamples);
  this->bufferManager.annGPUWorker->kernelBuilder->addUpdateKernels(numSamples);

  this->updateKernelsSetup = true;
}

//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::invalidateAllKernelFlags()
{
  this->predictKernelsSetup = false;
  this->trainingKernelsSetup = false;
  this->updateKernelsSetup = false;
}

//===================================================================================================================//
//-- addPropagateKernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addPropagateKernels()
{
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  Shape3D currentShape = this->coreConfig.inputShape;
  ulong convIdx = 0;
  ulong poolIdx = 0;

  for (ulong i = 0; i < cnnLayers.size(); i++) {
    const auto& layerConfig = cnnLayers[i];
    std::string layerStr = std::to_string(i);

    ulong inOffset = this->bufferManager.layerInfos[i].actvOffset;
    ulong outOffset = this->bufferManager.layerInfos[i + 1].actvOffset;

    switch (layerConfig.type) {
    case LayerType::CONV: {
      const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
      ulong nElements = conv.numFilters * outH * outW;

      std::string kernelId = "calculate_conv2d_layer" + layerStr;
      this->core->addKernel(kernelId, "calculate_conv2d", nElements, 0);
      this->core->template addArgument<T>(kernelId, "cnn_actvs");
      this->core->template addArgument<T>(kernelId, "cnn_filters");
      this->core->template addArgument<T>(kernelId, "cnn_biases");
      this->core->template addArgument<ulong>(kernelId, inOffset);
      this->core->template addArgument<ulong>(kernelId, outOffset);
      this->core->template addArgument<ulong>(kernelId, this->bufferManager.convInfos[convIdx].filterOffset);
      this->core->template addArgument<ulong>(kernelId, this->bufferManager.convInfos[convIdx].biasOffset);
      this->core->template addArgument<ulong>(kernelId, currentShape.c);
      this->core->template addArgument<ulong>(kernelId, currentShape.h);
      this->core->template addArgument<ulong>(kernelId, currentShape.w);
      this->core->template addArgument<ulong>(kernelId, conv.numFilters);
      this->core->template addArgument<ulong>(kernelId, conv.filterH);
      this->core->template addArgument<ulong>(kernelId, conv.filterW);
      this->core->template addArgument<ulong>(kernelId, conv.strideY);
      this->core->template addArgument<ulong>(kernelId, conv.strideX);
      this->core->template addArgument<ulong>(kernelId, padY);
      this->core->template addArgument<ulong>(kernelId, padX);
      this->core->template addArgument<ulong>(kernelId, outH);
      this->core->template addArgument<ulong>(kernelId, outW);

      currentShape = {conv.numFilters, outH, outW};
      convIdx++;
      break;
    }

    case LayerType::RELU: {
      ulong size = currentShape.size();
      std::string kernelId = "calculate_relu_layer" + layerStr;
      this->core->addKernel(kernelId, "calculate_relu", size, 0);
      this->core->template addArgument<T>(kernelId, "cnn_actvs");
      this->core->template addArgument<ulong>(kernelId, inOffset);
      this->core->template addArgument<ulong>(kernelId, outOffset);
      this->core->template addArgument<ulong>(kernelId, size);
      break;
    }

    case LayerType::POOL: {
      const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);
      ulong outH = (currentShape.h - pool.poolH) / pool.strideY + 1;
      ulong outW = (currentShape.w - pool.poolW) / pool.strideX + 1;
      ulong nElements = currentShape.c * outH * outW;

      std::string kernelId = "calculate_maxpool_layer" + layerStr;
      this->core->addKernel(kernelId, "calculate_maxpool", nElements, 0);
      this->core->template addArgument<T>(kernelId, "cnn_actvs");
      this->core->template addArgument<ulong>(kernelId, "cnn_pool_indices");
      this->core->template addArgument<ulong>(kernelId, inOffset);
      this->core->template addArgument<ulong>(kernelId, outOffset);
      this->core->template addArgument<ulong>(kernelId, this->bufferManager.poolInfos[poolIdx].indexOffset);
      this->core->template addArgument<ulong>(kernelId, currentShape.c);
      this->core->template addArgument<ulong>(kernelId, currentShape.h);
      this->core->template addArgument<ulong>(kernelId, currentShape.w);
      this->core->template addArgument<ulong>(kernelId, pool.poolH);
      this->core->template addArgument<ulong>(kernelId, pool.poolW);
      this->core->template addArgument<ulong>(kernelId, pool.strideY);
      this->core->template addArgument<ulong>(kernelId, pool.strideX);
      this->core->template addArgument<ulong>(kernelId, outH);
      this->core->template addArgument<ulong>(kernelId, outW);

      currentShape = {currentShape.c, outH, outW};
      poolIdx++;
      break;
    }

    case LayerType::FLATTEN: {
      break;
    }
    }
  }
}

//===================================================================================================================//
//-- addBackpropagateKernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addBackpropagateKernels()
{
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  ulong numLayers = cnnLayers.size();

  // Precompute shapes for each layer (propagate direction)
  std::vector<Shape3D> shapes(numLayers + 1);
  shapes[0] = this->coreConfig.inputShape;

  for (ulong i = 0; i < numLayers; i++) {
    Shape3D inShape = shapes[i];

    switch (cnnLayers[i].type) {
    case LayerType::CONV: {
      const auto& conv = std::get<ConvLayerConfig>(cnnLayers[i].config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (inShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (inShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
      shapes[i + 1] = {conv.numFilters, outH, outW};
      break;
    }

    case LayerType::RELU:
      shapes[i + 1] = inShape;
      break;
    case LayerType::POOL: {
      const auto& pool = std::get<PoolLayerConfig>(cnnLayers[i].config);
      ulong outH = (inShape.h - pool.poolH) / pool.strideY + 1;
      ulong outW = (inShape.w - pool.poolW) / pool.strideX + 1;
      shapes[i + 1] = {inShape.c, outH, outW};
      break;
    }

    case LayerType::FLATTEN:
      shapes[i + 1] = inShape;
      break;
    }
  }

  // Iterate through layers in reverse
  ulong convIdx = this->bufferManager.convInfos.size();
  ulong poolIdx = this->bufferManager.poolInfos.size();

  for (long i = static_cast<long>(numLayers) - 1; i >= 0; i--) {
    const auto& layerConfig = cnnLayers[static_cast<ulong>(i)];
    std::string layerStr = std::to_string(i);

    Shape3D inShape = shapes[static_cast<ulong>(i)];
    Shape3D outShape = shapes[static_cast<ulong>(i) + 1];

    ulong gradInOffset = this->bufferManager.layerInfos[static_cast<ulong>(i)].actvOffset;
    ulong gradOutOffset = this->bufferManager.layerInfos[static_cast<ulong>(i) + 1].actvOffset;
    ulong actvInOffset = this->bufferManager.layerInfos[static_cast<ulong>(i)].actvOffset;

    switch (layerConfig.type) {
    case LayerType::CONV: {
      convIdx--;
      const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = outShape.h;
      ulong outW = outShape.w;

      // calculate_dCost_dFilters
      ulong nFilterElems = this->bufferManager.convInfos[convIdx].numFilterElems;
      std::string filterId = "calculate_dCost_dFilters_layer" + layerStr;
      ulong filterLocalWS = 256;
      ulong filterGlobalWS = nFilterElems * filterLocalWS;
      this->core->addKernel(filterId, "calculate_dCost_dFilters", filterGlobalWS, 0, filterLocalWS);
      this->core->template addArgument<T>(filterId, "cnn_grads");
      this->core->template addArgument<T>(filterId, "cnn_actvs");
      this->core->template addArgument<T>(filterId, "cnn_dFilters");
      this->core->template addArgument<ulong>(filterId, gradOutOffset);
      this->core->template addArgument<ulong>(filterId, actvInOffset);
      this->core->template addArgument<ulong>(filterId, this->bufferManager.convInfos[convIdx].filterOffset);
      this->core->template addArgument<ulong>(filterId, inShape.c);
      this->core->template addArgument<ulong>(filterId, inShape.h);
      this->core->template addArgument<ulong>(filterId, inShape.w);
      this->core->template addArgument<ulong>(filterId, conv.numFilters);
      this->core->template addArgument<ulong>(filterId, conv.filterH);
      this->core->template addArgument<ulong>(filterId, conv.filterW);
      this->core->template addArgument<ulong>(filterId, conv.strideY);
      this->core->template addArgument<ulong>(filterId, conv.strideX);
      this->core->template addArgument<ulong>(filterId, padY);
      this->core->template addArgument<ulong>(filterId, padX);
      this->core->template addArgument<ulong>(filterId, outH);
      this->core->template addArgument<ulong>(filterId, outW);

      // calculate_dCost_dBiases
      std::string biasId = "calculate_dCost_dBiases_layer" + layerStr;
      ulong biasLocalWS = 256;
      ulong biasGlobalWS = conv.numFilters * biasLocalWS;
      this->core->addKernel(biasId, "calculate_dCost_dBiases", biasGlobalWS, 0, biasLocalWS);
      this->core->template addArgument<T>(biasId, "cnn_grads");
      this->core->template addArgument<T>(biasId, "cnn_dBiases");
      this->core->template addArgument<ulong>(biasId, gradOutOffset);
      this->core->template addArgument<ulong>(biasId, this->bufferManager.convInfos[convIdx].biasOffset);
      this->core->template addArgument<ulong>(biasId, conv.numFilters);
      this->core->template addArgument<ulong>(biasId, outH);
      this->core->template addArgument<ulong>(biasId, outW);

      // calculate_dCost_dInput (skip if first layer)
      if (i > 0) {
        ulong nInputElems = inShape.size();
        std::string inputId = "calculate_dCost_dInput_layer" + layerStr;
        this->core->addKernel(inputId, "calculate_dCost_dInput", nInputElems, 0);
        this->core->template addArgument<T>(inputId, "cnn_grads");
        this->core->template addArgument<T>(inputId, "cnn_filters");
        this->core->template addArgument<ulong>(inputId, gradOutOffset);
        this->core->template addArgument<ulong>(inputId, gradInOffset);
        this->core->template addArgument<ulong>(inputId, this->bufferManager.convInfos[convIdx].filterOffset);
        this->core->template addArgument<ulong>(inputId, inShape.c);
        this->core->template addArgument<ulong>(inputId, inShape.h);
        this->core->template addArgument<ulong>(inputId, inShape.w);
        this->core->template addArgument<ulong>(inputId, conv.numFilters);
        this->core->template addArgument<ulong>(inputId, conv.filterH);
        this->core->template addArgument<ulong>(inputId, conv.filterW);
        this->core->template addArgument<ulong>(inputId, conv.strideY);
        this->core->template addArgument<ulong>(inputId, conv.strideX);
        this->core->template addArgument<ulong>(inputId, padY);
        this->core->template addArgument<ulong>(inputId, padX);
        this->core->template addArgument<ulong>(inputId, outH);
        this->core->template addArgument<ulong>(inputId, outW);
      }

      break;
    }

    case LayerType::RELU: {
      ulong size = inShape.size();
      std::string kernelId = "calculate_dCost_dRelu_layer" + layerStr;
      this->core->addKernel(kernelId, "calculate_dCost_dRelu", size, 0);
      this->core->template addArgument<T>(kernelId, "cnn_grads");
      this->core->template addArgument<T>(kernelId, "cnn_actvs");
      this->core->template addArgument<ulong>(kernelId, gradInOffset);
      this->core->template addArgument<ulong>(kernelId, gradOutOffset);
      this->core->template addArgument<ulong>(kernelId, actvInOffset);
      this->core->template addArgument<ulong>(kernelId, size);
      break;
    }

    case LayerType::POOL: {
      poolIdx--;

      ulong inSize = inShape.size();
      std::string zeroId = "zero_pool_grad_layer" + layerStr;
      this->core->addKernel(zeroId, "zero_buffer", inSize, 0);
      this->core->template addArgument<T>(zeroId, "cnn_grads");
      this->core->template addArgument<ulong>(zeroId, gradInOffset);
      this->core->template addArgument<ulong>(zeroId, inSize);

      ulong outSize = outShape.size();
      std::string poolId = "calculate_dCost_dMaxpool_layer" + layerStr;
      this->core->addKernel(poolId, "calculate_dCost_dMaxpool", outSize, 0);
      this->core->template addArgument<T>(poolId, "cnn_grads");
      this->core->template addArgument<ulong>(poolId, "cnn_pool_indices");
      this->core->template addArgument<ulong>(poolId, gradOutOffset);
      this->core->template addArgument<ulong>(poolId, this->bufferManager.poolInfos[poolIdx].indexOffset);
      this->core->template addArgument<ulong>(poolId, outSize);
      break;
    }

    case LayerType::FLATTEN: {
      break;
    }
    }
  }
}

//===================================================================================================================//
//-- addCopyBridgeKernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addCopyBridgeKernels()
{
  ulong lastLayerIdx = this->bufferManager.layerInfos.size() - 1;
  ulong cnnOutputOffset = this->bufferManager.layerInfos[lastLayerIdx].actvOffset;

  this->core->addKernel("copy_cnn_to_ann", this->bufferManager.flattenSize, 0);
  this->core->template addArgument<T>("copy_cnn_to_ann", "cnn_actvs");
  this->core->template addArgument<T>("copy_cnn_to_ann", "actvs");
  this->core->template addArgument<ulong>("copy_cnn_to_ann", cnnOutputOffset);
  this->core->template addArgument<ulong>("copy_cnn_to_ann", this->bufferManager.flattenSize);
}

//===================================================================================================================//
//-- addCNNAccumulateKernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addCNNAccumulateKernels()
{
  if (this->bufferManager.totalFilterSize > 0) {
    this->core->addKernel("accumulate_gradients_filters", "accumulate_gradients", this->bufferManager.totalFilterSize,
                          0);
    this->core->template addArgument<T>("accumulate_gradients_filters", "cnn_accum_dFilters");
    this->core->template addArgument<T>("accumulate_gradients_filters", "cnn_dFilters");
    this->core->template addArgument<ulong>("accumulate_gradients_filters", static_cast<ulong>(0));
    this->core->template addArgument<ulong>("accumulate_gradients_filters", this->bufferManager.totalFilterSize);
  }

  if (this->bufferManager.totalBiasSize > 0) {
    this->core->addKernel("accumulate_gradients_biases", "accumulate_gradients", this->bufferManager.totalBiasSize, 0);
    this->core->template addArgument<T>("accumulate_gradients_biases", "cnn_accum_dBiases");
    this->core->template addArgument<T>("accumulate_gradients_biases", "cnn_dBiases");
    this->core->template addArgument<ulong>("accumulate_gradients_biases", static_cast<ulong>(0));
    this->core->template addArgument<ulong>("accumulate_gradients_biases", this->bufferManager.totalBiasSize);
  }
}

//===================================================================================================================//
//-- addCNNUpdateKernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addCNNUpdateKernels(ulong numSamples)
{
  if (this->coreConfig.trainingConfig.optimizer.type == OptimizerType::ADAM) {
    const auto& opt = this->coreConfig.trainingConfig.optimizer;
    this->adam_t++;

    float bc1 = 1.0f - std::pow(static_cast<float>(opt.beta1), static_cast<float>(this->adam_t));
    float bc2 = 1.0f - std::pow(static_cast<float>(opt.beta2), static_cast<float>(this->adam_t));

    if (this->bufferManager.totalFilterSize > 0) {
      this->core->addKernel("update_parameters_filters", "update_parameters_adam", this->bufferManager.totalFilterSize,
                            0);
      this->core->template addArgument<T>("update_parameters_filters", "cnn_filters");
      this->core->template addArgument<T>("update_parameters_filters", "cnn_accum_dFilters");
      this->core->template addArgument<T>("update_parameters_filters", "cnn_adam_m_filters");
      this->core->template addArgument<T>("update_parameters_filters", "cnn_adam_v_filters");
      this->core->template addArgument<ulong>("update_parameters_filters", static_cast<ulong>(0));
      this->core->template addArgument<ulong>("update_parameters_filters", this->bufferManager.totalFilterSize);
      this->core->template addArgument<ulong>("update_parameters_filters", numSamples);
      this->core->template addArgument<float>("update_parameters_filters",
                                              static_cast<float>(this->coreConfig.trainingConfig.learningRate));
      this->core->template addArgument<float>("update_parameters_filters", static_cast<float>(opt.beta1));
      this->core->template addArgument<float>("update_parameters_filters", static_cast<float>(opt.beta2));
      this->core->template addArgument<float>("update_parameters_filters", static_cast<float>(opt.epsilon));
      this->core->template addArgument<float>("update_parameters_filters", bc1);
      this->core->template addArgument<float>("update_parameters_filters", bc2);
    }

    if (this->bufferManager.totalBiasSize > 0) {
      this->core->addKernel("update_parameters_biases", "update_parameters_adam", this->bufferManager.totalBiasSize, 0);
      this->core->template addArgument<T>("update_parameters_biases", "cnn_biases");
      this->core->template addArgument<T>("update_parameters_biases", "cnn_accum_dBiases");
      this->core->template addArgument<T>("update_parameters_biases", "cnn_adam_m_biases");
      this->core->template addArgument<T>("update_parameters_biases", "cnn_adam_v_biases");
      this->core->template addArgument<ulong>("update_parameters_biases", static_cast<ulong>(0));
      this->core->template addArgument<ulong>("update_parameters_biases", this->bufferManager.totalBiasSize);
      this->core->template addArgument<ulong>("update_parameters_biases", numSamples);
      this->core->template addArgument<float>("update_parameters_biases",
                                              static_cast<float>(this->coreConfig.trainingConfig.learningRate));
      this->core->template addArgument<float>("update_parameters_biases", static_cast<float>(opt.beta1));
      this->core->template addArgument<float>("update_parameters_biases", static_cast<float>(opt.beta2));
      this->core->template addArgument<float>("update_parameters_biases", static_cast<float>(opt.epsilon));
      this->core->template addArgument<float>("update_parameters_biases", bc1);
      this->core->template addArgument<float>("update_parameters_biases", bc2);
    }
  } else {
    // SGD
    if (this->bufferManager.totalFilterSize > 0) {
      this->core->addKernel("update_parameters_filters", "update_parameters", this->bufferManager.totalFilterSize, 0);
      this->core->template addArgument<T>("update_parameters_filters", "cnn_filters");
      this->core->template addArgument<T>("update_parameters_filters", "cnn_accum_dFilters");
      this->core->template addArgument<ulong>("update_parameters_filters", static_cast<ulong>(0));
      this->core->template addArgument<ulong>("update_parameters_filters", this->bufferManager.totalFilterSize);
      this->core->template addArgument<ulong>("update_parameters_filters", numSamples);
      this->core->template addArgument<float>("update_parameters_filters",
                                              static_cast<float>(this->coreConfig.trainingConfig.learningRate));
    }

    if (this->bufferManager.totalBiasSize > 0) {
      this->core->addKernel("update_parameters_biases", "update_parameters", this->bufferManager.totalBiasSize, 0);
      this->core->template addArgument<T>("update_parameters_biases", "cnn_biases");
      this->core->template addArgument<T>("update_parameters_biases", "cnn_accum_dBiases");
      this->core->template addArgument<ulong>("update_parameters_biases", static_cast<ulong>(0));
      this->core->template addArgument<ulong>("update_parameters_biases", this->bufferManager.totalBiasSize);
      this->core->template addArgument<ulong>("update_parameters_biases", numSamples);
      this->core->template addArgument<float>("update_parameters_biases",
                                              static_cast<float>(this->coreConfig.trainingConfig.learningRate));
    }
  }
}

//===================================================================================================================//
// Explicit template instantiations.
//===================================================================================================================//

template class CNN::GPUKernelBuilder<int>;
template class CNN::GPUKernelBuilder<float>;
template class CNN::GPUKernelBuilder<double>;