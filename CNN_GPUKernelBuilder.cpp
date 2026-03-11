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
  this->addPropagateKernels(true);
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
void GPUKernelBuilder<T>::addPropagateKernels(bool training)
{
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  Shape3D currentShape = this->coreConfig.inputShape;
  ulong convIdx = 0;
  ulong poolIdx = 0;
  ulong normIdx = 0;

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

      if (pool.poolType == PoolTypeEnum::MAX) {
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
      } else {
        // Average pooling — no pool_indices needed
        std::string kernelId = "calculate_avgpool_layer" + layerStr;
        this->core->addKernel(kernelId, "calculate_avgpool", nElements, 0);
        this->core->template addArgument<T>(kernelId, "cnn_actvs");
        this->core->template addArgument<ulong>(kernelId, inOffset);
        this->core->template addArgument<ulong>(kernelId, outOffset);
        this->core->template addArgument<ulong>(kernelId, currentShape.c);
        this->core->template addArgument<ulong>(kernelId, currentShape.h);
        this->core->template addArgument<ulong>(kernelId, currentShape.w);
        this->core->template addArgument<ulong>(kernelId, pool.poolH);
        this->core->template addArgument<ulong>(kernelId, pool.poolW);
        this->core->template addArgument<ulong>(kernelId, pool.strideY);
        this->core->template addArgument<ulong>(kernelId, pool.strideX);
        this->core->template addArgument<ulong>(kernelId, outH);
        this->core->template addArgument<ulong>(kernelId, outW);
      }

      currentShape = {currentShape.c, outH, outW};
      poolIdx++;
      break;
    }

    case LayerType::INSTANCENORM: {
      const auto& bn = std::get<NormLayerConfig>(layerConfig.config);
      ulong size = currentShape.size();
      ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;

      // Always compute per-sample spatial mean/var before normalizing.
      // Since each sample is normalised independently over (H,W) during training,
      // inference must do the same to stay consistent.
      {
        ulong localWS = 256;
        ulong meanGlobalWS = currentShape.c * localWS;

        std::string meanId = "norm_compute_mean_layer" + layerStr;
        this->core->addKernel(meanId, "norm_compute_mean", meanGlobalWS, 0, localWS);
        this->core->template addArgument<T>(meanId, "cnn_actvs");
        this->core->template addArgument<T>(meanId, "cnn_norm_batch_mean");
        this->core->template addArgument<ulong>(meanId, inOffset);
        this->core->template addArgument<ulong>(meanId, normParamOffset);
        this->core->template addArgument<ulong>(meanId, currentShape.c);
        this->core->template addArgument<ulong>(meanId, currentShape.h);
        this->core->template addArgument<ulong>(meanId, currentShape.w);
        this->core->template addArgument<ulong>(meanId, static_cast<ulong>(1));
        this->core->template addArgument<ulong>(meanId, static_cast<ulong>(0));

        std::string varId = "norm_compute_var_layer" + layerStr;
        this->core->addKernel(varId, "norm_compute_var", meanGlobalWS, 0, localWS);
        this->core->template addArgument<T>(varId, "cnn_actvs");
        this->core->template addArgument<T>(varId, "cnn_norm_batch_mean");
        this->core->template addArgument<T>(varId, "cnn_norm_batch_var");
        this->core->template addArgument<ulong>(varId, inOffset);
        this->core->template addArgument<ulong>(varId, normParamOffset);
        this->core->template addArgument<ulong>(varId, currentShape.c);
        this->core->template addArgument<ulong>(varId, currentShape.h);
        this->core->template addArgument<ulong>(varId, currentShape.w);
        this->core->template addArgument<ulong>(varId, static_cast<ulong>(1));
        this->core->template addArgument<ulong>(varId, static_cast<ulong>(0));
      }

      // Always use per-sample spatial stats for normalization
      std::string meanBuf = "cnn_norm_batch_mean";
      std::string varBuf = "cnn_norm_batch_var";

      std::string normId = "norm_normalize_layer" + layerStr;
      this->core->addKernel(normId, "norm_normalize", size, 0);
      this->core->template addArgument<T>(normId, "cnn_actvs");
      this->core->template addArgument<T>(normId, "cnn_norm_xnorm");
      this->core->template addArgument<T>(normId, "cnn_norm_gamma");
      this->core->template addArgument<T>(normId, "cnn_norm_beta");
      this->core->template addArgument<T>(normId, meanBuf);
      this->core->template addArgument<T>(normId, varBuf);
      this->core->template addArgument<ulong>(normId, inOffset);
      this->core->template addArgument<ulong>(normId, outOffset);
      this->core->template addArgument<ulong>(normId, inOffset); // xnorm offset
      this->core->template addArgument<ulong>(normId, normParamOffset);
      this->core->template addArgument<ulong>(normId, currentShape.c);
      this->core->template addArgument<ulong>(normId, currentShape.h);
      this->core->template addArgument<ulong>(normId, currentShape.w);
      this->core->template addArgument<ulong>(normId, static_cast<ulong>(1));
      this->core->template addArgument<ulong>(normId, static_cast<ulong>(0));
      this->core->template addArgument<float>(normId, bn.epsilon);

      normIdx++;
      break;
    }

    case LayerType::BATCHNORM: {
      const auto& bn = std::get<NormLayerConfig>(layerConfig.config);
      ulong size = currentShape.size();
      ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;

      if (training) {
        // Training: compute per-sample spatial mean/var (N=1, sampleStride=0)
        ulong localWS = 256;
        ulong meanGlobalWS = currentShape.c * localWS;

        std::string meanId = "norm_compute_mean_layer" + layerStr;
        this->core->addKernel(meanId, "norm_compute_mean", meanGlobalWS, 0, localWS);
        this->core->template addArgument<T>(meanId, "cnn_actvs");
        this->core->template addArgument<T>(meanId, "cnn_norm_batch_mean");
        this->core->template addArgument<ulong>(meanId, inOffset);
        this->core->template addArgument<ulong>(meanId, normParamOffset);
        this->core->template addArgument<ulong>(meanId, currentShape.c);
        this->core->template addArgument<ulong>(meanId, currentShape.h);
        this->core->template addArgument<ulong>(meanId, currentShape.w);
        this->core->template addArgument<ulong>(meanId, static_cast<ulong>(1));
        this->core->template addArgument<ulong>(meanId, static_cast<ulong>(0));

        std::string varId = "norm_compute_var_layer" + layerStr;
        this->core->addKernel(varId, "norm_compute_var", meanGlobalWS, 0, localWS);
        this->core->template addArgument<T>(varId, "cnn_actvs");
        this->core->template addArgument<T>(varId, "cnn_norm_batch_mean");
        this->core->template addArgument<T>(varId, "cnn_norm_batch_var");
        this->core->template addArgument<ulong>(varId, inOffset);
        this->core->template addArgument<ulong>(varId, normParamOffset);
        this->core->template addArgument<ulong>(varId, currentShape.c);
        this->core->template addArgument<ulong>(varId, currentShape.h);
        this->core->template addArgument<ulong>(varId, currentShape.w);
        this->core->template addArgument<ulong>(varId, static_cast<ulong>(1));
        this->core->template addArgument<ulong>(varId, static_cast<ulong>(0));

        std::string normId = "norm_normalize_layer" + layerStr;
        this->core->addKernel(normId, "norm_normalize", size, 0);
        this->core->template addArgument<T>(normId, "cnn_actvs");
        this->core->template addArgument<T>(normId, "cnn_norm_xnorm");
        this->core->template addArgument<T>(normId, "cnn_norm_gamma");
        this->core->template addArgument<T>(normId, "cnn_norm_beta");
        this->core->template addArgument<T>(normId, "cnn_norm_batch_mean");
        this->core->template addArgument<T>(normId, "cnn_norm_batch_var");
        this->core->template addArgument<ulong>(normId, inOffset);
        this->core->template addArgument<ulong>(normId, outOffset);
        this->core->template addArgument<ulong>(normId, inOffset);
        this->core->template addArgument<ulong>(normId, normParamOffset);
        this->core->template addArgument<ulong>(normId, currentShape.c);
        this->core->template addArgument<ulong>(normId, currentShape.h);
        this->core->template addArgument<ulong>(normId, currentShape.w);
        this->core->template addArgument<ulong>(normId, static_cast<ulong>(1));
        this->core->template addArgument<ulong>(normId, static_cast<ulong>(0));
        this->core->template addArgument<float>(normId, bn.epsilon);
      } else {
        // Inference: use running stats (N=1, sampleStride=0)
        std::string normId = "norm_normalize_layer" + layerStr;
        this->core->addKernel(normId, "norm_normalize", size, 0);
        this->core->template addArgument<T>(normId, "cnn_actvs");
        this->core->template addArgument<T>(normId, "cnn_norm_xnorm");
        this->core->template addArgument<T>(normId, "cnn_norm_gamma");
        this->core->template addArgument<T>(normId, "cnn_norm_beta");
        this->core->template addArgument<T>(normId, "cnn_norm_running_mean");
        this->core->template addArgument<T>(normId, "cnn_norm_running_var");
        this->core->template addArgument<ulong>(normId, inOffset);
        this->core->template addArgument<ulong>(normId, outOffset);
        this->core->template addArgument<ulong>(normId, inOffset);
        this->core->template addArgument<ulong>(normId, normParamOffset);
        this->core->template addArgument<ulong>(normId, currentShape.c);
        this->core->template addArgument<ulong>(normId, currentShape.h);
        this->core->template addArgument<ulong>(normId, currentShape.w);
        this->core->template addArgument<ulong>(normId, static_cast<ulong>(1));
        this->core->template addArgument<ulong>(normId, static_cast<ulong>(0));
        this->core->template addArgument<float>(normId, bn.epsilon);
      }

      normIdx++;
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

    case LayerType::INSTANCENORM:
      shapes[i + 1] = inShape;
      break;
    case LayerType::BATCHNORM:
      shapes[i + 1] = inShape;
      break;
    case LayerType::FLATTEN:
      shapes[i + 1] = inShape;
      break;
    }
  }

  // Iterate through layers in reverse
  ulong convIdx = this->bufferManager.convInfos.size();
  ulong poolIdx = this->bufferManager.poolInfos.size();
  ulong normIdx = this->bufferManager.normInfos.size();

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
      const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);

      ulong inSize = inShape.size();
      std::string zeroId = "zero_pool_grad_layer" + layerStr;
      this->core->addKernel(zeroId, "zero_buffer", inSize, 0);
      this->core->template addArgument<T>(zeroId, "cnn_grads");
      this->core->template addArgument<ulong>(zeroId, gradInOffset);
      this->core->template addArgument<ulong>(zeroId, inSize);

      ulong outSize = outShape.size();

      if (pool.poolType == PoolTypeEnum::MAX) {
        std::string poolId = "calculate_dCost_dMaxpool_layer" + layerStr;
        this->core->addKernel(poolId, "calculate_dCost_dMaxpool", outSize, 0);
        this->core->template addArgument<T>(poolId, "cnn_grads");
        this->core->template addArgument<ulong>(poolId, "cnn_pool_indices");
        this->core->template addArgument<ulong>(poolId, gradOutOffset);
        this->core->template addArgument<ulong>(poolId, this->bufferManager.poolInfos[poolIdx].indexOffset);
        this->core->template addArgument<ulong>(poolId, outSize);
      } else {
        std::string poolId = "calculate_dCost_dAvgpool_layer" + layerStr;
        this->core->addKernel(poolId, "calculate_dCost_dAvgpool", outSize, 0);
        this->core->template addArgument<T>(poolId, "cnn_grads");
        this->core->template addArgument<ulong>(poolId, gradInOffset);
        this->core->template addArgument<ulong>(poolId, gradOutOffset);
        this->core->template addArgument<ulong>(poolId, inShape.c);
        this->core->template addArgument<ulong>(poolId, inShape.h);
        this->core->template addArgument<ulong>(poolId, inShape.w);
        this->core->template addArgument<ulong>(poolId, pool.poolH);
        this->core->template addArgument<ulong>(poolId, pool.poolW);
        this->core->template addArgument<ulong>(poolId, pool.strideY);
        this->core->template addArgument<ulong>(poolId, pool.strideX);
        this->core->template addArgument<ulong>(poolId, outShape.h);
        this->core->template addArgument<ulong>(poolId, outShape.w);
      }

      break;
    }

    case LayerType::INSTANCENORM:
    case LayerType::BATCHNORM: {
      normIdx--;
      const auto& bn = std::get<NormLayerConfig>(layerConfig.config);
      ulong size = inShape.size();
      ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;

      // dGamma and dBeta (N=1, sampleStride=0 for per-sample backprop)
      ulong localWS = 256;
      ulong dgGlobalWS = inShape.c * localWS;
      std::string dgId = "norm_dGammaBeta_layer" + layerStr;
      this->core->addKernel(dgId, "norm_dGammaBeta", dgGlobalWS, 0, localWS);
      this->core->template addArgument<T>(dgId, "cnn_grads");
      this->core->template addArgument<T>(dgId, "cnn_norm_xnorm");
      this->core->template addArgument<T>(dgId, "cnn_norm_dGamma");
      this->core->template addArgument<T>(dgId, "cnn_norm_dBeta");
      this->core->template addArgument<ulong>(dgId, gradOutOffset);
      this->core->template addArgument<ulong>(dgId, actvInOffset);
      this->core->template addArgument<ulong>(dgId, normParamOffset);
      this->core->template addArgument<ulong>(dgId, inShape.c);
      this->core->template addArgument<ulong>(dgId, inShape.h);
      this->core->template addArgument<ulong>(dgId, inShape.w);
      this->core->template addArgument<ulong>(dgId, static_cast<ulong>(1));
      this->core->template addArgument<ulong>(dgId, static_cast<ulong>(0));

      // dInput (N=1, sampleStride=0 for per-sample backprop)
      std::string diId = "norm_dInput_layer" + layerStr;
      this->core->addKernel(diId, "norm_dInput", size, 0);
      this->core->template addArgument<T>(diId, "cnn_grads");
      this->core->template addArgument<T>(diId, "cnn_norm_xnorm");
      this->core->template addArgument<T>(diId, "cnn_norm_gamma");
      this->core->template addArgument<T>(diId, "cnn_norm_dGamma");
      this->core->template addArgument<T>(diId, "cnn_norm_dBeta");
      this->core->template addArgument<T>(diId, "cnn_norm_batch_var");
      this->core->template addArgument<ulong>(diId, gradInOffset);
      this->core->template addArgument<ulong>(diId, gradOutOffset);
      this->core->template addArgument<ulong>(diId, actvInOffset);
      this->core->template addArgument<ulong>(diId, normParamOffset);
      this->core->template addArgument<ulong>(diId, inShape.c);
      this->core->template addArgument<ulong>(diId, inShape.h);
      this->core->template addArgument<ulong>(diId, inShape.w);
      this->core->template addArgument<ulong>(diId, static_cast<ulong>(1));
      this->core->template addArgument<ulong>(diId, static_cast<ulong>(0));
      this->core->template addArgument<float>(diId, bn.epsilon);
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

  if (this->bufferManager.totalNormParamSize > 0) {
    this->core->addKernel("accumulate_gradients_norm_gamma", "accumulate_gradients",
                          this->bufferManager.totalNormParamSize, 0);
    this->core->template addArgument<T>("accumulate_gradients_norm_gamma", "cnn_accum_norm_dGamma");
    this->core->template addArgument<T>("accumulate_gradients_norm_gamma", "cnn_norm_dGamma");
    this->core->template addArgument<ulong>("accumulate_gradients_norm_gamma", static_cast<ulong>(0));
    this->core->template addArgument<ulong>("accumulate_gradients_norm_gamma", this->bufferManager.totalNormParamSize);

    this->core->addKernel("accumulate_gradients_norm_beta", "accumulate_gradients",
                          this->bufferManager.totalNormParamSize, 0);
    this->core->template addArgument<T>("accumulate_gradients_norm_beta", "cnn_accum_norm_dBeta");
    this->core->template addArgument<T>("accumulate_gradients_norm_beta", "cnn_norm_dBeta");
    this->core->template addArgument<ulong>("accumulate_gradients_norm_beta", static_cast<ulong>(0));
    this->core->template addArgument<ulong>("accumulate_gradients_norm_beta", this->bufferManager.totalNormParamSize);

    // Accumulate batch mean/var across samples for running stats update
    this->core->addKernel("accumulate_norm_batch_mean", "accumulate_gradients", this->bufferManager.totalNormParamSize,
                          0);
    this->core->template addArgument<T>("accumulate_norm_batch_mean", "cnn_accum_norm_batch_mean");
    this->core->template addArgument<T>("accumulate_norm_batch_mean", "cnn_norm_batch_mean");
    this->core->template addArgument<ulong>("accumulate_norm_batch_mean", static_cast<ulong>(0));
    this->core->template addArgument<ulong>("accumulate_norm_batch_mean", this->bufferManager.totalNormParamSize);

    this->core->addKernel("accumulate_norm_batch_var", "accumulate_gradients", this->bufferManager.totalNormParamSize,
                          0);
    this->core->template addArgument<T>("accumulate_norm_batch_var", "cnn_accum_norm_batch_var");
    this->core->template addArgument<T>("accumulate_norm_batch_var", "cnn_norm_batch_var");
    this->core->template addArgument<ulong>("accumulate_norm_batch_var", static_cast<ulong>(0));
    this->core->template addArgument<ulong>("accumulate_norm_batch_var", this->bufferManager.totalNormParamSize);
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

    if (this->bufferManager.totalNormParamSize > 0) {
      this->core->addKernel("update_parameters_norm_gamma", "update_parameters_adam",
                            this->bufferManager.totalNormParamSize, 0);
      this->core->template addArgument<T>("update_parameters_norm_gamma", "cnn_norm_gamma");
      this->core->template addArgument<T>("update_parameters_norm_gamma", "cnn_accum_norm_dGamma");
      this->core->template addArgument<T>("update_parameters_norm_gamma", "cnn_adam_m_norm_gamma");
      this->core->template addArgument<T>("update_parameters_norm_gamma", "cnn_adam_v_norm_gamma");
      this->core->template addArgument<ulong>("update_parameters_norm_gamma", static_cast<ulong>(0));
      this->core->template addArgument<ulong>("update_parameters_norm_gamma", this->bufferManager.totalNormParamSize);
      this->core->template addArgument<ulong>("update_parameters_norm_gamma", numSamples);
      this->core->template addArgument<float>("update_parameters_norm_gamma",
                                              static_cast<float>(this->coreConfig.trainingConfig.learningRate));
      this->core->template addArgument<float>("update_parameters_norm_gamma", static_cast<float>(opt.beta1));
      this->core->template addArgument<float>("update_parameters_norm_gamma", static_cast<float>(opt.beta2));
      this->core->template addArgument<float>("update_parameters_norm_gamma", static_cast<float>(opt.epsilon));
      this->core->template addArgument<float>("update_parameters_norm_gamma", bc1);
      this->core->template addArgument<float>("update_parameters_norm_gamma", bc2);

      this->core->addKernel("update_parameters_norm_beta", "update_parameters_adam",
                            this->bufferManager.totalNormParamSize, 0);
      this->core->template addArgument<T>("update_parameters_norm_beta", "cnn_norm_beta");
      this->core->template addArgument<T>("update_parameters_norm_beta", "cnn_accum_norm_dBeta");
      this->core->template addArgument<T>("update_parameters_norm_beta", "cnn_adam_m_norm_beta");
      this->core->template addArgument<T>("update_parameters_norm_beta", "cnn_adam_v_norm_beta");
      this->core->template addArgument<ulong>("update_parameters_norm_beta", static_cast<ulong>(0));
      this->core->template addArgument<ulong>("update_parameters_norm_beta", this->bufferManager.totalNormParamSize);
      this->core->template addArgument<ulong>("update_parameters_norm_beta", numSamples);
      this->core->template addArgument<float>("update_parameters_norm_beta",
                                              static_cast<float>(this->coreConfig.trainingConfig.learningRate));
      this->core->template addArgument<float>("update_parameters_norm_beta", static_cast<float>(opt.beta1));
      this->core->template addArgument<float>("update_parameters_norm_beta", static_cast<float>(opt.beta2));
      this->core->template addArgument<float>("update_parameters_norm_beta", static_cast<float>(opt.epsilon));
      this->core->template addArgument<float>("update_parameters_norm_beta", bc1);
      this->core->template addArgument<float>("update_parameters_norm_beta", bc2);
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

    if (this->bufferManager.totalNormParamSize > 0) {
      this->core->addKernel("update_parameters_norm_gamma", "update_parameters", this->bufferManager.totalNormParamSize,
                            0);
      this->core->template addArgument<T>("update_parameters_norm_gamma", "cnn_norm_gamma");
      this->core->template addArgument<T>("update_parameters_norm_gamma", "cnn_accum_norm_dGamma");
      this->core->template addArgument<ulong>("update_parameters_norm_gamma", static_cast<ulong>(0));
      this->core->template addArgument<ulong>("update_parameters_norm_gamma", this->bufferManager.totalNormParamSize);
      this->core->template addArgument<ulong>("update_parameters_norm_gamma", numSamples);
      this->core->template addArgument<float>("update_parameters_norm_gamma",
                                              static_cast<float>(this->coreConfig.trainingConfig.learningRate));

      this->core->addKernel("update_parameters_norm_beta", "update_parameters", this->bufferManager.totalNormParamSize,
                            0);
      this->core->template addArgument<T>("update_parameters_norm_beta", "cnn_norm_beta");
      this->core->template addArgument<T>("update_parameters_norm_beta", "cnn_accum_norm_dBeta");
      this->core->template addArgument<ulong>("update_parameters_norm_beta", static_cast<ulong>(0));
      this->core->template addArgument<ulong>("update_parameters_norm_beta", this->bufferManager.totalNormParamSize);
      this->core->template addArgument<ulong>("update_parameters_norm_beta", numSamples);
      this->core->template addArgument<float>("update_parameters_norm_beta",
                                              static_cast<float>(this->coreConfig.trainingConfig.learningRate));
    }
  }

  // Running stats update (same for both Adam and SGD)
  if (this->bufferManager.totalNormParamSize > 0) {
    // Build a list of norm layer types
    std::vector<LayerType> normLayerTypes;

    for (const auto& layerConfig : this->coreConfig.layersConfig.cnnLayers) {
      if (layerConfig.type == LayerType::INSTANCENORM || layerConfig.type == LayerType::BATCHNORM) {
        normLayerTypes.push_back(layerConfig.type);
      }
    }

    // Get momentum from first norm layer config
    float momentum = 0.1f;

    for (const auto& layerConfig : this->coreConfig.layersConfig.cnnLayers) {
      if (layerConfig.type == LayerType::INSTANCENORM || layerConfig.type == LayerType::BATCHNORM) {
        const auto& bn = std::get<NormLayerConfig>(layerConfig.config);
        momentum = bn.momentum;
        break;
      }
    }

    for (ulong b = 0; b < this->bufferManager.normInfos.size(); b++) {
      // Skip BATCHNORM running stats update when using batch norm training path
      // (batch norm path updates running stats directly with batch-wide mean/var)
      if (this->skipBNRunningStatsInUpdate && normLayerTypes[b] == LayerType::BATCHNORM) {
        continue;
      }

      ulong normParamOffset = this->bufferManager.normInfos[b].paramOffset;
      ulong numChannels = this->bufferManager.normInfos[b].numChannels;
      std::string kernelId = "norm_update_running_stats_" + std::to_string(b);
      this->core->addKernel(kernelId, "norm_update_running_stats", numChannels, 0);
      this->core->template addArgument<T>(kernelId, "cnn_norm_running_mean");
      this->core->template addArgument<T>(kernelId, "cnn_norm_running_var");
      this->core->template addArgument<T>(kernelId, "cnn_accum_norm_batch_mean");
      this->core->template addArgument<T>(kernelId, "cnn_accum_norm_batch_var");
      this->core->template addArgument<ulong>(kernelId, normParamOffset);
      this->core->template addArgument<ulong>(kernelId, numChannels);
      this->core->template addArgument<float>(kernelId, momentum);
      this->core->template addArgument<ulong>(kernelId, numSamples);
    }
  }
}

//===================================================================================================================//
//-- Batch-norm-aware: propagate kernels for one sample in batch buffers --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addBatchPropagateKernelsForSample(ulong sampleIdx, ulong layerStart, ulong layerEnd)
{
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  ulong sampleStride = this->bufferManager.totalActvSize;
  ulong sampleOffset = sampleIdx * sampleStride;

  // Compute shape at layerStart
  Shape3D currentShape = this->coreConfig.inputShape;
  ulong convIdx = 0;
  ulong poolIdx = 0;
  ulong normIdx = 0;

  for (ulong i = 0; i < layerStart; i++) {
    switch (cnnLayers[i].type) {
    case LayerType::CONV: {
      const auto& conv = std::get<ConvLayerConfig>(cnnLayers[i].config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
      currentShape = {conv.numFilters, outH, outW};
      convIdx++;
      break;
    }

    case LayerType::POOL: {
      const auto& pool = std::get<PoolLayerConfig>(cnnLayers[i].config);
      ulong outH = (currentShape.h - pool.poolH) / pool.strideY + 1;
      ulong outW = (currentShape.w - pool.poolW) / pool.strideX + 1;
      currentShape = {currentShape.c, outH, outW};
      poolIdx++;
      break;
    }

    case LayerType::INSTANCENORM:
    case LayerType::BATCHNORM:
      normIdx++;
      break;
    default:
      break;
    }
  }

  std::string sampleStr = std::to_string(sampleIdx);

  for (ulong i = layerStart; i < layerEnd; i++) {
    const auto& layerConfig = cnnLayers[i];
    std::string layerStr = std::to_string(i);
    std::string kernelSuffix = "_s" + sampleStr + "_l" + layerStr;

    ulong inOffset = sampleOffset + this->bufferManager.layerInfos[i].actvOffset;
    ulong outOffset = sampleOffset + this->bufferManager.layerInfos[i + 1].actvOffset;

    switch (layerConfig.type) {
    case LayerType::CONV: {
      const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
      ulong nElements = conv.numFilters * outH * outW;

      std::string kernelId = "batch_conv2d" + kernelSuffix;
      this->core->addKernel(kernelId, "calculate_conv2d", nElements, 0);
      this->core->template addArgument<T>(kernelId, "cnn_batch_actvs");
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
      std::string kernelId = "batch_relu" + kernelSuffix;
      this->core->addKernel(kernelId, "calculate_relu", size, 0);
      this->core->template addArgument<T>(kernelId, "cnn_batch_actvs");
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
      ulong poolIdxOffset =
        sampleIdx * this->bufferManager.totalPoolIndexSize + this->bufferManager.poolInfos[poolIdx].indexOffset;

      if (pool.poolType == PoolTypeEnum::MAX) {
        std::string kernelId = "batch_maxpool" + kernelSuffix;
        this->core->addKernel(kernelId, "calculate_maxpool", nElements, 0);
        this->core->template addArgument<T>(kernelId, "cnn_batch_actvs");
        this->core->template addArgument<ulong>(kernelId, "cnn_batch_pool_indices");
        this->core->template addArgument<ulong>(kernelId, inOffset);
        this->core->template addArgument<ulong>(kernelId, outOffset);
        this->core->template addArgument<ulong>(kernelId, poolIdxOffset);
        this->core->template addArgument<ulong>(kernelId, currentShape.c);
        this->core->template addArgument<ulong>(kernelId, currentShape.h);
        this->core->template addArgument<ulong>(kernelId, currentShape.w);
        this->core->template addArgument<ulong>(kernelId, pool.poolH);
        this->core->template addArgument<ulong>(kernelId, pool.poolW);
        this->core->template addArgument<ulong>(kernelId, pool.strideY);
        this->core->template addArgument<ulong>(kernelId, pool.strideX);
        this->core->template addArgument<ulong>(kernelId, outH);
        this->core->template addArgument<ulong>(kernelId, outW);
      } else {
        std::string kernelId = "batch_avgpool" + kernelSuffix;
        this->core->addKernel(kernelId, "calculate_avgpool", nElements, 0);
        this->core->template addArgument<T>(kernelId, "cnn_batch_actvs");
        this->core->template addArgument<ulong>(kernelId, inOffset);
        this->core->template addArgument<ulong>(kernelId, outOffset);
        this->core->template addArgument<ulong>(kernelId, currentShape.c);
        this->core->template addArgument<ulong>(kernelId, currentShape.h);
        this->core->template addArgument<ulong>(kernelId, currentShape.w);
        this->core->template addArgument<ulong>(kernelId, pool.poolH);
        this->core->template addArgument<ulong>(kernelId, pool.poolW);
        this->core->template addArgument<ulong>(kernelId, pool.strideY);
        this->core->template addArgument<ulong>(kernelId, pool.strideX);
        this->core->template addArgument<ulong>(kernelId, outH);
        this->core->template addArgument<ulong>(kernelId, outW);
      }

      currentShape = {currentShape.c, outH, outW};
      poolIdx++;
      break;
    }

    case LayerType::INSTANCENORM: {
      const auto& bn = std::get<NormLayerConfig>(layerConfig.config);
      ulong size = currentShape.size();
      ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;

      // Per-sample spatial mean/var (N=1, sampleStride=0)
      ulong localWS = 256;
      ulong meanGlobalWS = currentShape.c * localWS;

      std::string meanId = "batch_norm_mean" + kernelSuffix;
      this->core->addKernel(meanId, "norm_compute_mean", meanGlobalWS, 0, localWS);
      this->core->template addArgument<T>(meanId, "cnn_batch_actvs");
      this->core->template addArgument<T>(meanId, "cnn_norm_batch_mean");
      this->core->template addArgument<ulong>(meanId, inOffset);
      this->core->template addArgument<ulong>(meanId, normParamOffset);
      this->core->template addArgument<ulong>(meanId, currentShape.c);
      this->core->template addArgument<ulong>(meanId, currentShape.h);
      this->core->template addArgument<ulong>(meanId, currentShape.w);
      this->core->template addArgument<ulong>(meanId, static_cast<ulong>(1));
      this->core->template addArgument<ulong>(meanId, static_cast<ulong>(0));

      std::string varId = "batch_norm_var" + kernelSuffix;
      this->core->addKernel(varId, "norm_compute_var", meanGlobalWS, 0, localWS);
      this->core->template addArgument<T>(varId, "cnn_batch_actvs");
      this->core->template addArgument<T>(varId, "cnn_norm_batch_mean");
      this->core->template addArgument<T>(varId, "cnn_norm_batch_var");
      this->core->template addArgument<ulong>(varId, inOffset);
      this->core->template addArgument<ulong>(varId, normParamOffset);
      this->core->template addArgument<ulong>(varId, currentShape.c);
      this->core->template addArgument<ulong>(varId, currentShape.h);
      this->core->template addArgument<ulong>(varId, currentShape.w);
      this->core->template addArgument<ulong>(varId, static_cast<ulong>(1));
      this->core->template addArgument<ulong>(varId, static_cast<ulong>(0));

      std::string normId = "batch_norm_normalize" + kernelSuffix;
      this->core->addKernel(normId, "norm_normalize", size, 0);
      this->core->template addArgument<T>(normId, "cnn_batch_actvs");
      this->core->template addArgument<T>(normId, "cnn_batch_xnorm");
      this->core->template addArgument<T>(normId, "cnn_norm_gamma");
      this->core->template addArgument<T>(normId, "cnn_norm_beta");
      this->core->template addArgument<T>(normId, "cnn_norm_batch_mean");
      this->core->template addArgument<T>(normId, "cnn_norm_batch_var");
      this->core->template addArgument<ulong>(normId, inOffset);
      this->core->template addArgument<ulong>(normId, outOffset);
      this->core->template addArgument<ulong>(normId, sampleOffset + this->bufferManager.layerInfos[i].actvOffset);
      this->core->template addArgument<ulong>(normId, normParamOffset);
      this->core->template addArgument<ulong>(normId, currentShape.c);
      this->core->template addArgument<ulong>(normId, currentShape.h);
      this->core->template addArgument<ulong>(normId, currentShape.w);
      this->core->template addArgument<ulong>(normId, static_cast<ulong>(1));
      this->core->template addArgument<ulong>(normId, static_cast<ulong>(0));
      this->core->template addArgument<float>(normId, bn.epsilon);

      normIdx++;
      break;
    }

    case LayerType::BATCHNORM: {
      // Skip — handled by addBatchNormForwardKernels
      normIdx++;
      break;
    }

    case LayerType::FLATTEN: {
      break;
    }
    }
  }
}

//===================================================================================================================//
//-- Batch-norm-aware: batch-wide BN forward kernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addBatchNormForwardKernels(ulong layerIdx, ulong batchSize)
{
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  const auto& bn = std::get<NormLayerConfig>(cnnLayers[layerIdx].config);
  std::string layerStr = std::to_string(layerIdx);

  // Compute shape and normIdx at this layer
  Shape3D currentShape = this->coreConfig.inputShape;
  ulong normIdx = 0;

  for (ulong i = 0; i < layerIdx; i++) {
    switch (cnnLayers[i].type) {
    case LayerType::CONV: {
      const auto& conv = std::get<ConvLayerConfig>(cnnLayers[i].config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
      currentShape = {conv.numFilters, outH, outW};
      break;
    }

    case LayerType::POOL: {
      const auto& pool = std::get<PoolLayerConfig>(cnnLayers[i].config);
      ulong outH = (currentShape.h - pool.poolH) / pool.strideY + 1;
      ulong outW = (currentShape.w - pool.poolW) / pool.strideX + 1;
      currentShape = {currentShape.c, outH, outW};
      break;
    }

    case LayerType::INSTANCENORM:
    case LayerType::BATCHNORM:
      normIdx++;
      break;
    default:
      break;
    }
  }

  ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;
  ulong actvInOffset = this->bufferManager.layerInfos[layerIdx].actvOffset;
  ulong actvOutOffset = this->bufferManager.layerInfos[layerIdx + 1].actvOffset;
  ulong sampleStride = this->bufferManager.totalActvSize;

  // Batch-wide mean
  ulong localWS = 256;
  ulong meanGlobalWS = currentShape.c * localWS;

  std::string meanId = "bn_batch_mean_l" + layerStr;
  this->core->addKernel(meanId, "norm_compute_mean", meanGlobalWS, 0, localWS);
  this->core->template addArgument<T>(meanId, "cnn_batch_actvs");
  this->core->template addArgument<T>(meanId, "cnn_norm_batch_mean");
  this->core->template addArgument<ulong>(meanId, actvInOffset);
  this->core->template addArgument<ulong>(meanId, normParamOffset);
  this->core->template addArgument<ulong>(meanId, currentShape.c);
  this->core->template addArgument<ulong>(meanId, currentShape.h);
  this->core->template addArgument<ulong>(meanId, currentShape.w);
  this->core->template addArgument<ulong>(meanId, batchSize);
  this->core->template addArgument<ulong>(meanId, sampleStride);

  // Batch-wide variance
  std::string varId = "bn_batch_var_l" + layerStr;
  this->core->addKernel(varId, "norm_compute_var", meanGlobalWS, 0, localWS);
  this->core->template addArgument<T>(varId, "cnn_batch_actvs");
  this->core->template addArgument<T>(varId, "cnn_norm_batch_mean");
  this->core->template addArgument<T>(varId, "cnn_norm_batch_var");
  this->core->template addArgument<ulong>(varId, actvInOffset);
  this->core->template addArgument<ulong>(varId, normParamOffset);
  this->core->template addArgument<ulong>(varId, currentShape.c);
  this->core->template addArgument<ulong>(varId, currentShape.h);
  this->core->template addArgument<ulong>(varId, currentShape.w);
  this->core->template addArgument<ulong>(varId, batchSize);
  this->core->template addArgument<ulong>(varId, sampleStride);

  // Batch-wide normalize (all N samples at once)
  ulong totalElements = batchSize * currentShape.size();
  std::string normId = "bn_batch_normalize_l" + layerStr;
  this->core->addKernel(normId, "norm_normalize", totalElements, 0);
  this->core->template addArgument<T>(normId, "cnn_batch_actvs");
  this->core->template addArgument<T>(normId, "cnn_batch_xnorm");
  this->core->template addArgument<T>(normId, "cnn_norm_gamma");
  this->core->template addArgument<T>(normId, "cnn_norm_beta");
  this->core->template addArgument<T>(normId, "cnn_norm_batch_mean");
  this->core->template addArgument<T>(normId, "cnn_norm_batch_var");
  this->core->template addArgument<ulong>(normId, actvInOffset);
  this->core->template addArgument<ulong>(normId, actvOutOffset);
  this->core->template addArgument<ulong>(normId, actvInOffset); // xnorm offset
  this->core->template addArgument<ulong>(normId, normParamOffset);
  this->core->template addArgument<ulong>(normId, currentShape.c);
  this->core->template addArgument<ulong>(normId, currentShape.h);
  this->core->template addArgument<ulong>(normId, currentShape.w);
  this->core->template addArgument<ulong>(normId, batchSize);
  this->core->template addArgument<ulong>(normId, sampleStride);
  this->core->template addArgument<float>(normId, bn.epsilon);
}

//===================================================================================================================//
//-- Batch-norm-aware: backpropagate kernels for one sample in batch buffers --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addBatchBackpropagateKernelsForSample(ulong sampleIdx, ulong layerStart, ulong layerEnd)
{
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  ulong numLayers = cnnLayers.size();
  ulong sampleStride = this->bufferManager.totalActvSize;
  ulong sampleOffset = sampleIdx * sampleStride;

  // Precompute shapes for all layers
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

    case LayerType::POOL: {
      const auto& pool = std::get<PoolLayerConfig>(cnnLayers[i].config);
      ulong outH = (inShape.h - pool.poolH) / pool.strideY + 1;
      ulong outW = (inShape.w - pool.poolW) / pool.strideX + 1;
      shapes[i + 1] = {inShape.c, outH, outW};
      break;
    }

    default:
      shapes[i + 1] = inShape;
      break;
    }
  }

  // Count conv/pool/norm indices up to layerEnd
  ulong convIdx = this->bufferManager.convInfos.size();
  ulong poolIdx = this->bufferManager.poolInfos.size();
  ulong normIdx = this->bufferManager.normInfos.size();

  // Recount from 0 to get correct indices
  convIdx = 0;
  poolIdx = 0;
  normIdx = 0;

  for (ulong i = 0; i < numLayers; i++) {
    if (i >= layerEnd)
      break;
    switch (cnnLayers[i].type) {
    case LayerType::CONV:
      convIdx++;
      break;
    case LayerType::POOL:
      poolIdx++;
      break;
    case LayerType::INSTANCENORM:
    case LayerType::BATCHNORM:
      normIdx++;
      break;
    default:
      break;
    }
  }

  std::string sampleStr = std::to_string(sampleIdx);

  // Iterate in reverse from layerEnd-1 down to layerStart
  for (long i = static_cast<long>(layerEnd) - 1; i >= static_cast<long>(layerStart); i--) {
    const auto& layerConfig = cnnLayers[static_cast<ulong>(i)];
    std::string layerStr = std::to_string(i);
    std::string kernelSuffix = "_s" + sampleStr + "_l" + layerStr;

    Shape3D inShape = shapes[static_cast<ulong>(i)];
    Shape3D outShape = shapes[static_cast<ulong>(i) + 1];

    ulong gradInOffset = sampleOffset + this->bufferManager.layerInfos[static_cast<ulong>(i)].actvOffset;
    ulong gradOutOffset = sampleOffset + this->bufferManager.layerInfos[static_cast<ulong>(i) + 1].actvOffset;
    ulong actvInOffset = sampleOffset + this->bufferManager.layerInfos[static_cast<ulong>(i)].actvOffset;

    switch (layerConfig.type) {
    case LayerType::CONV: {
      convIdx--;
      const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = outShape.h;
      ulong outW = outShape.w;

      // dFilters
      ulong nFilterElems = this->bufferManager.convInfos[convIdx].numFilterElems;
      std::string filterId = "batch_dFilters" + kernelSuffix;
      ulong filterLocalWS = 256;
      ulong filterGlobalWS = nFilterElems * filterLocalWS;
      this->core->addKernel(filterId, "calculate_dCost_dFilters", filterGlobalWS, 0, filterLocalWS);
      this->core->template addArgument<T>(filterId, "cnn_batch_grads");
      this->core->template addArgument<T>(filterId, "cnn_batch_actvs");
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

      // dBiases
      std::string biasId = "batch_dBiases" + kernelSuffix;
      ulong biasLocalWS = 256;
      ulong biasGlobalWS = conv.numFilters * biasLocalWS;
      this->core->addKernel(biasId, "calculate_dCost_dBiases", biasGlobalWS, 0, biasLocalWS);
      this->core->template addArgument<T>(biasId, "cnn_batch_grads");
      this->core->template addArgument<T>(biasId, "cnn_dBiases");
      this->core->template addArgument<ulong>(biasId, gradOutOffset);
      this->core->template addArgument<ulong>(biasId, this->bufferManager.convInfos[convIdx].biasOffset);
      this->core->template addArgument<ulong>(biasId, conv.numFilters);
      this->core->template addArgument<ulong>(biasId, outH);
      this->core->template addArgument<ulong>(biasId, outW);

      // dInput (skip if first layer)
      if (i > 0) {
        ulong nInputElems = inShape.size();
        std::string inputId = "batch_dInput" + kernelSuffix;
        this->core->addKernel(inputId, "calculate_dCost_dInput", nInputElems, 0);
        this->core->template addArgument<T>(inputId, "cnn_batch_grads");
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
      std::string kernelId = "batch_dRelu" + kernelSuffix;
      this->core->addKernel(kernelId, "calculate_dCost_dRelu", size, 0);
      this->core->template addArgument<T>(kernelId, "cnn_batch_grads");
      this->core->template addArgument<T>(kernelId, "cnn_batch_actvs");
      this->core->template addArgument<ulong>(kernelId, gradInOffset);
      this->core->template addArgument<ulong>(kernelId, gradOutOffset);
      this->core->template addArgument<ulong>(kernelId, actvInOffset);
      this->core->template addArgument<ulong>(kernelId, size);
      break;
    }

    case LayerType::POOL: {
      poolIdx--;
      const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);
      ulong inSize = inShape.size();
      ulong outSize = outShape.size();
      ulong poolIdxOffset =
        sampleIdx * this->bufferManager.totalPoolIndexSize + this->bufferManager.poolInfos[poolIdx].indexOffset;

      std::string zeroId = "batch_zero_pool" + kernelSuffix;
      this->core->addKernel(zeroId, "zero_buffer", inSize, 0);
      this->core->template addArgument<T>(zeroId, "cnn_batch_grads");
      this->core->template addArgument<ulong>(zeroId, gradInOffset);
      this->core->template addArgument<ulong>(zeroId, inSize);

      if (pool.poolType == PoolTypeEnum::MAX) {
        std::string poolId = "batch_dMaxpool" + kernelSuffix;
        this->core->addKernel(poolId, "calculate_dCost_dMaxpool", outSize, 0);
        this->core->template addArgument<T>(poolId, "cnn_batch_grads");
        this->core->template addArgument<ulong>(poolId, "cnn_batch_pool_indices");
        this->core->template addArgument<ulong>(poolId, gradOutOffset);
        this->core->template addArgument<ulong>(poolId, poolIdxOffset);
        this->core->template addArgument<ulong>(poolId, outSize);
      } else {
        std::string poolId = "batch_dAvgpool" + kernelSuffix;
        this->core->addKernel(poolId, "calculate_dCost_dAvgpool", outSize, 0);
        this->core->template addArgument<T>(poolId, "cnn_batch_grads");
        this->core->template addArgument<ulong>(poolId, gradInOffset);
        this->core->template addArgument<ulong>(poolId, gradOutOffset);
        this->core->template addArgument<ulong>(poolId, inShape.c);
        this->core->template addArgument<ulong>(poolId, inShape.h);
        this->core->template addArgument<ulong>(poolId, inShape.w);
        this->core->template addArgument<ulong>(poolId, pool.poolH);
        this->core->template addArgument<ulong>(poolId, pool.poolW);
        this->core->template addArgument<ulong>(poolId, pool.strideY);
        this->core->template addArgument<ulong>(poolId, pool.strideX);
        this->core->template addArgument<ulong>(poolId, outShape.h);
        this->core->template addArgument<ulong>(poolId, outShape.w);
      }

      break;
    }

    case LayerType::INSTANCENORM: {
      normIdx--;
      const auto& bn = std::get<NormLayerConfig>(layerConfig.config);
      ulong size = inShape.size();
      ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;

      ulong localWS = 256;
      ulong dgGlobalWS = inShape.c * localWS;
      std::string dgId = "batch_norm_dGammaBeta" + kernelSuffix;
      this->core->addKernel(dgId, "norm_dGammaBeta", dgGlobalWS, 0, localWS);
      this->core->template addArgument<T>(dgId, "cnn_batch_grads");
      this->core->template addArgument<T>(dgId, "cnn_batch_xnorm");
      this->core->template addArgument<T>(dgId, "cnn_norm_dGamma");
      this->core->template addArgument<T>(dgId, "cnn_norm_dBeta");
      this->core->template addArgument<ulong>(dgId, gradOutOffset);
      this->core->template addArgument<ulong>(dgId, actvInOffset);
      this->core->template addArgument<ulong>(dgId, normParamOffset);
      this->core->template addArgument<ulong>(dgId, inShape.c);
      this->core->template addArgument<ulong>(dgId, inShape.h);
      this->core->template addArgument<ulong>(dgId, inShape.w);
      this->core->template addArgument<ulong>(dgId, static_cast<ulong>(1));
      this->core->template addArgument<ulong>(dgId, static_cast<ulong>(0));

      std::string diId = "batch_norm_dInput" + kernelSuffix;
      this->core->addKernel(diId, "norm_dInput", size, 0);
      this->core->template addArgument<T>(diId, "cnn_batch_grads");
      this->core->template addArgument<T>(diId, "cnn_batch_xnorm");
      this->core->template addArgument<T>(diId, "cnn_norm_gamma");
      this->core->template addArgument<T>(diId, "cnn_norm_dGamma");
      this->core->template addArgument<T>(diId, "cnn_norm_dBeta");
      this->core->template addArgument<T>(diId, "cnn_norm_batch_var");
      this->core->template addArgument<ulong>(diId, gradInOffset);
      this->core->template addArgument<ulong>(diId, gradOutOffset);
      this->core->template addArgument<ulong>(diId, actvInOffset);
      this->core->template addArgument<ulong>(diId, normParamOffset);
      this->core->template addArgument<ulong>(diId, inShape.c);
      this->core->template addArgument<ulong>(diId, inShape.h);
      this->core->template addArgument<ulong>(diId, inShape.w);
      this->core->template addArgument<ulong>(diId, static_cast<ulong>(1));
      this->core->template addArgument<ulong>(diId, static_cast<ulong>(0));
      this->core->template addArgument<float>(diId, bn.epsilon);
      break;
    }

    case LayerType::BATCHNORM: {
      // Skip — handled by addBatchNormBackwardKernels
      normIdx--;
      break;
    }

    case LayerType::FLATTEN: {
      break;
    }
    }
  }
}

//===================================================================================================================//
//-- Batch-norm-aware: batch-wide BN backward kernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addBatchNormBackwardKernels(ulong layerIdx, ulong batchSize)
{
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  const auto& bn = std::get<NormLayerConfig>(cnnLayers[layerIdx].config);
  std::string layerStr = std::to_string(layerIdx);

  // Compute shape and normIdx at this layer
  Shape3D currentShape = this->coreConfig.inputShape;
  ulong normIdx = 0;

  for (ulong i = 0; i < layerIdx; i++) {
    switch (cnnLayers[i].type) {
    case LayerType::CONV: {
      const auto& conv = std::get<ConvLayerConfig>(cnnLayers[i].config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
      currentShape = {conv.numFilters, outH, outW};
      break;
    }

    case LayerType::POOL: {
      const auto& pool = std::get<PoolLayerConfig>(cnnLayers[i].config);
      ulong outH = (currentShape.h - pool.poolH) / pool.strideY + 1;
      ulong outW = (currentShape.w - pool.poolW) / pool.strideX + 1;
      currentShape = {currentShape.c, outH, outW};
      break;
    }

    case LayerType::INSTANCENORM:
    case LayerType::BATCHNORM:
      normIdx++;
      break;
    default:
      break;
    }
  }

  ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;
  ulong gradOutOffset = this->bufferManager.layerInfos[layerIdx + 1].actvOffset;
  ulong gradInOffset = this->bufferManager.layerInfos[layerIdx].actvOffset;
  ulong xnormOffset = this->bufferManager.layerInfos[layerIdx].actvOffset;
  ulong sampleStride = this->bufferManager.totalActvSize;

  // Batch-wide dGamma/dBeta (N=batchSize, sampleStride=sampleStride)
  ulong localWS = 256;
  ulong dgGlobalWS = currentShape.c * localWS;
  std::string dgId = "bn_batch_dGammaBeta_l" + layerStr;
  this->core->addKernel(dgId, "norm_dGammaBeta", dgGlobalWS, 0, localWS);
  this->core->template addArgument<T>(dgId, "cnn_batch_grads");
  this->core->template addArgument<T>(dgId, "cnn_batch_xnorm");
  this->core->template addArgument<T>(dgId, "cnn_norm_dGamma");
  this->core->template addArgument<T>(dgId, "cnn_norm_dBeta");
  this->core->template addArgument<ulong>(dgId, gradOutOffset);
  this->core->template addArgument<ulong>(dgId, xnormOffset);
  this->core->template addArgument<ulong>(dgId, normParamOffset);
  this->core->template addArgument<ulong>(dgId, currentShape.c);
  this->core->template addArgument<ulong>(dgId, currentShape.h);
  this->core->template addArgument<ulong>(dgId, currentShape.w);
  this->core->template addArgument<ulong>(dgId, batchSize);
  this->core->template addArgument<ulong>(dgId, sampleStride);

  // Batch-wide dInput (all N samples at once)
  ulong totalElements = batchSize * currentShape.size();
  std::string diId = "bn_batch_dInput_l" + layerStr;
  this->core->addKernel(diId, "norm_dInput", totalElements, 0);
  this->core->template addArgument<T>(diId, "cnn_batch_grads");
  this->core->template addArgument<T>(diId, "cnn_batch_xnorm");
  this->core->template addArgument<T>(diId, "cnn_norm_gamma");
  this->core->template addArgument<T>(diId, "cnn_norm_dGamma");
  this->core->template addArgument<T>(diId, "cnn_norm_dBeta");
  this->core->template addArgument<T>(diId, "cnn_norm_batch_var");
  this->core->template addArgument<ulong>(diId, gradInOffset);
  this->core->template addArgument<ulong>(diId, gradOutOffset);
  this->core->template addArgument<ulong>(diId, xnormOffset);
  this->core->template addArgument<ulong>(diId, normParamOffset);
  this->core->template addArgument<ulong>(diId, currentShape.c);
  this->core->template addArgument<ulong>(diId, currentShape.h);
  this->core->template addArgument<ulong>(diId, currentShape.w);
  this->core->template addArgument<ulong>(diId, batchSize);
  this->core->template addArgument<ulong>(diId, sampleStride);
  this->core->template addArgument<float>(diId, bn.epsilon);
}

//===================================================================================================================//
//-- Batch-norm-aware: copy bridge from batch buffer to ANN --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addBatchCopyBridgeKernels(ulong sampleIdx)
{
  ulong sampleStride = this->bufferManager.totalActvSize;
  ulong lastLayerIdx = this->bufferManager.layerInfos.size() - 1;
  ulong cnnOutputOffset = sampleIdx * sampleStride + this->bufferManager.layerInfos[lastLayerIdx].actvOffset;

  std::string kernelId = "batch_copy_cnn_to_ann_s" + std::to_string(sampleIdx);
  this->core->addKernel(kernelId, "copy_cnn_to_ann", this->bufferManager.flattenSize, 0);
  this->core->template addArgument<T>(kernelId, "cnn_batch_actvs");
  this->core->template addArgument<T>(kernelId, "actvs");
  this->core->template addArgument<ulong>(kernelId, cnnOutputOffset);
  this->core->template addArgument<ulong>(kernelId, this->bufferManager.flattenSize);
}

//===================================================================================================================//
//-- Batch-norm-aware: reverse bridge from ANN grads to batch gradient buffer --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addBatchReverseBridgeKernels(ulong sampleIdx)
{
  ulong sampleStride = this->bufferManager.totalActvSize;
  ulong lastLayerIdx = this->bufferManager.layerInfos.size() - 1;
  ulong cnnOutputOffset = sampleIdx * sampleStride + this->bufferManager.layerInfos[lastLayerIdx].actvOffset;

  std::string kernelId = "batch_copy_ann_grad_to_cnn_s" + std::to_string(sampleIdx);
  this->core->addKernel(kernelId, "copy_ann_grad_to_cnn", this->bufferManager.flattenSize, 0);
  this->core->template addArgument<T>(kernelId, "dCost_dActvs");
  this->core->template addArgument<T>(kernelId, "cnn_batch_grads");
  this->core->template addArgument<ulong>(kernelId, cnnOutputOffset);
  this->core->template addArgument<ulong>(kernelId, this->bufferManager.flattenSize);
}

//===================================================================================================================//
//-- Batch-norm-aware: accumulate CNN gradients for one sample --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addBatchCNNAccumulateKernelsForSample(ulong sampleIdx, ulong layerStart, ulong layerEnd)
{
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  std::string sampleStr = std::to_string(sampleIdx);

  // Only accumulate gradients for conv layers within [layerStart, layerEnd).
  // Each conv layer's gradients occupy a specific region in cnn_dFilters/cnn_dBiases.
  ulong convIdx = 0;

  for (ulong i = 0; i < layerEnd; i++) {
    if (cnnLayers[i].type == LayerType::CONV) {
      if (i >= layerStart) {
        ulong filterOffset = this->bufferManager.convInfos[convIdx].filterOffset;
        ulong numFilterElems = this->bufferManager.convInfos[convIdx].numFilterElems;
        ulong biasOffset = this->bufferManager.convInfos[convIdx].biasOffset;
        ulong numBiases = this->bufferManager.convInfos[convIdx].numBiases;

        std::string kernelId = "batch_accum_filters_s" + sampleStr + "_c" + std::to_string(convIdx);
        this->core->addKernel(kernelId, "accumulate_gradients", numFilterElems, 0);
        this->core->template addArgument<T>(kernelId, "cnn_accum_dFilters");
        this->core->template addArgument<T>(kernelId, "cnn_dFilters");
        this->core->template addArgument<ulong>(kernelId, filterOffset);
        this->core->template addArgument<ulong>(kernelId, numFilterElems);

        std::string biasKernelId = "batch_accum_biases_s" + sampleStr + "_c" + std::to_string(convIdx);
        this->core->addKernel(biasKernelId, "accumulate_gradients", numBiases, 0);
        this->core->template addArgument<T>(biasKernelId, "cnn_accum_dBiases");
        this->core->template addArgument<T>(biasKernelId, "cnn_dBiases");
        this->core->template addArgument<ulong>(biasKernelId, biasOffset);
        this->core->template addArgument<ulong>(biasKernelId, numBiases);
      }

      convIdx++;
    }
  }
}

//===================================================================================================================//
//-- Batch-norm-aware: running stats update using batch-wide mean/var --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addBatchNormRunningStatsUpdate(ulong batchSize)
{
  if (this->bufferManager.totalNormParamSize == 0)
    return;

  // Get momentum from first BN layer config
  float momentum = 0.1f;

  for (const auto& layerConfig : this->coreConfig.layersConfig.cnnLayers) {
    if (layerConfig.type == LayerType::BATCHNORM) {
      const auto& bn = std::get<NormLayerConfig>(layerConfig.config);
      momentum = bn.momentum;
      break;
    }
  }

  // For BatchNorm layers, the batch mean/var are already the true batch statistics
  // (computed by norm_compute_mean/var with N=batchSize). We update running stats directly.
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  ulong normIdx = 0;

  for (ulong i = 0; i < cnnLayers.size(); i++) {
    if (cnnLayers[i].type == LayerType::BATCHNORM) {
      ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;
      ulong numChannels = this->bufferManager.normInfos[normIdx].numChannels;
      std::string kernelId = "bn_update_running_stats_" + std::to_string(normIdx);
      this->core->addKernel(kernelId, "norm_update_running_stats", numChannels, 0);
      this->core->template addArgument<T>(kernelId, "cnn_norm_running_mean");
      this->core->template addArgument<T>(kernelId, "cnn_norm_running_var");
      this->core->template addArgument<T>(kernelId, "cnn_norm_batch_mean");
      this->core->template addArgument<T>(kernelId, "cnn_norm_batch_var");
      this->core->template addArgument<ulong>(kernelId, normParamOffset);
      this->core->template addArgument<ulong>(kernelId, numChannels);
      this->core->template addArgument<float>(kernelId, momentum);
      // For batch-wide stats, numSamples=1 since mean/var are already the batch average
      this->core->template addArgument<ulong>(kernelId, static_cast<ulong>(1));
    }

    if (cnnLayers[i].type == LayerType::INSTANCENORM || cnnLayers[i].type == LayerType::BATCHNORM) {
      normIdx++;
    }
  }
}

//===================================================================================================================//
// Explicit template instantiations.
//===================================================================================================================//

template class CNN::GPUKernelBuilder<int>;
template class CNN::GPUKernelBuilder<float>;
template class CNN::GPUKernelBuilder<double>;