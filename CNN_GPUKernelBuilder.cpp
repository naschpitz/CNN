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

  // Predict uses sample 0 slot; bufferManager.batchSize == 1 for inference (uses running stats for BN)
  this->addPropagateKernels(0);
  this->addCopyBridgeKernels(0);
  this->bufferManager.annGPUWorker->kernelBuilder->addPropagateKernels();

  this->predictKernelsSetup = true;
}

//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::setupBatchForwardKernels()
{
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  ulong batchSize = this->bufferManager.batchSize;

  // CNN forward for all N samples, layer-by-layer
  for (ulong s = 0; s < batchSize; s++) {
    this->addPropagateKernels(s);
  }
}

//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::setupPerSampleANNKernels(ulong sampleIdx)
{
  this->core->clearKernels();

  // Bridge: copy CNN output for this sample to ANN input
  this->addCopyBridgeKernels(sampleIdx);

  // ANN forward (training mode)
  this->bufferManager.annGPUWorker->kernelBuilder->addPropagateKernels(true);

  // ANN backward (with input gradients for reverse bridge)
  this->bufferManager.annGPUWorker->kernelBuilder->addBackpropagateKernels(true);

  // Reverse bridge: copy ANN input gradients to CNN gradient buffer at sample-specific offset
  ulong sampleStride = this->bufferManager.totalActvSize;
  ulong lastLayerIdx = this->bufferManager.layerInfos.size() - 1;
  ulong cnnOutputOffset = sampleIdx * sampleStride + this->bufferManager.layerInfos[lastLayerIdx].actvOffset;
  std::string kernelId = "copy_ann_grad_to_cnn_s" + std::to_string(sampleIdx);
  this->core->addKernel(kernelId, "copy_ann_grad_to_cnn", this->bufferManager.flattenSize, 0);
  this->core->template addArgument<T>(kernelId, "dCost_dActvs");
  this->core->template addArgument<T>(kernelId, "cnn_grads");
  this->core->template addArgument<ulong>(kernelId, cnnOutputOffset);
  this->core->template addArgument<ulong>(kernelId, this->bufferManager.flattenSize);

  // ANN accumulate
  this->bufferManager.annGPUWorker->kernelBuilder->addAccumulateKernels();

  // Loss: compute weighted loss on GPU and accumulate into accum_loss buffer
  ulong outputActvOffset = this->bufferManager.annGPUWorker->bufferManager->getOutputActvOffset();
  ulong numOutputNeurons = this->bufferManager.annGPUWorker->bufferManager->getNumOutputNeurons();
  std::string lossId = "calculate_sample_loss_s" + std::to_string(sampleIdx);
  this->core->addKernel(lossId, "calculate_sample_loss", 1, 0);
  this->core->template addArgument<T>(lossId, "actvs");
  this->core->template addArgument<T>(lossId, "outputs");
  this->core->template addArgument<T>(lossId, "lossWeights");
  this->core->template addArgument<T>(lossId, "accum_loss");
  this->core->template addArgument<ulong>(lossId, outputActvOffset);
  this->core->template addArgument<ulong>(lossId, numOutputNeurons);
  this->core->template addArgument<ulong>(lossId, static_cast<ulong>(this->coreConfig.costFunctionConfig.type));
}

//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::setupBatchBackwardKernels()
{
  this->core->clearKernels();

  ulong batchSize = this->bufferManager.batchSize;

  // CNN backward for all N samples. Each sample's backward pass computes per-sample
  // gradients into cnn_dFilters/cnn_dBiases, which are then accumulated into the
  // accum buffers before the next sample overwrites them.
  // BatchNorm dGammaBeta is batch-wide (only added once for sample 0).
  for (ulong s = 0; s < batchSize; s++) {
    this->addBackpropagateKernels(s);
    this->addCNNPerSampleAccumulateKernels(s);
  }

  // Norm mean/var accumulation for running stats (once per batch, not per sample)
  this->addCNNNormStatsAccumulateKernels();
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
  this->updateKernelsSetup = false;
}

//===================================================================================================================//
//-- addPropagateKernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addPropagateKernels(ulong sampleIdx)
{
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  Shape3D currentShape = this->coreConfig.inputShape;
  ulong convIdx = 0;
  ulong poolIdx = 0;
  ulong normIdx = 0;
  ulong batchSize = this->bufferManager.batchSize;

  ulong sampleStride = this->bufferManager.totalActvSize;
  ulong poolSampleStride = this->bufferManager.totalPoolIndexSize;
  std::string sStr = "s" + std::to_string(sampleIdx);

  for (ulong i = 0; i < cnnLayers.size(); i++) {
    const auto& layerConfig = cnnLayers[i];
    std::string layerStr = std::to_string(i);

    ulong inOffset = sampleIdx * sampleStride + this->bufferManager.layerInfos[i].actvOffset;
    ulong outOffset = sampleIdx * sampleStride + this->bufferManager.layerInfos[i + 1].actvOffset;

    switch (layerConfig.type) {
    case LayerType::CONV: {
      const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
      ulong nElements = conv.numFilters * outH * outW;

      std::string kernelId = "calculate_conv2d_" + sStr + "_layer" + layerStr;
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
      std::string kernelId = "calculate_relu_" + sStr + "_layer" + layerStr;
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
      ulong poolIndexOffset = sampleIdx * poolSampleStride + this->bufferManager.poolInfos[poolIdx].indexOffset;

      if (pool.poolType == PoolTypeEnum::MAX) {
        std::string kernelId = "calculate_maxpool_" + sStr + "_layer" + layerStr;
        this->core->addKernel(kernelId, "calculate_maxpool", nElements, 0);
        this->core->template addArgument<T>(kernelId, "cnn_actvs");
        this->core->template addArgument<ulong>(kernelId, "cnn_pool_indices");
        this->core->template addArgument<ulong>(kernelId, inOffset);
        this->core->template addArgument<ulong>(kernelId, outOffset);
        this->core->template addArgument<ulong>(kernelId, poolIndexOffset);
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
        std::string kernelId = "calculate_avgpool_" + sStr + "_layer" + layerStr;
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
      // InstanceNorm: per-image spatial stats (N=1, H×W per channel)
      const auto& norm = std::get<NormLayerConfig>(layerConfig.config);
      ulong size = currentShape.size();
      ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;
      ulong layerActvOffset = this->bufferManager.layerInfos[i].actvOffset;
      ulong localWS = 256;
      ulong meanGlobalWS = currentShape.c * localWS;

      std::string meanId = "calculate_norm_mean_" + sStr + "_layer" + layerStr;
      this->core->addKernel(meanId, "calculate_norm_mean", meanGlobalWS, 0, localWS);
      this->core->template addArgument<T>(meanId, "cnn_actvs");
      this->core->template addArgument<T>(meanId, "cnn_norm_batch_mean");
      this->core->template addArgument<ulong>(meanId, normParamOffset);
      this->core->template addArgument<ulong>(meanId, currentShape.c);
      this->core->template addArgument<ulong>(meanId, currentShape.h);
      this->core->template addArgument<ulong>(meanId, currentShape.w);
      this->core->template addArgument<ulong>(meanId, static_cast<ulong>(1)); // N = 1
      this->core->template addArgument<ulong>(meanId, static_cast<ulong>(0)); // sample_stride = 0
      this->core->template addArgument<ulong>(meanId, inOffset); // actv_layer_offset = inOffset

      std::string varId = "calculate_norm_var_" + sStr + "_layer" + layerStr;
      this->core->addKernel(varId, "calculate_norm_var", meanGlobalWS, 0, localWS);
      this->core->template addArgument<T>(varId, "cnn_actvs");
      this->core->template addArgument<T>(varId, "cnn_norm_batch_mean");
      this->core->template addArgument<T>(varId, "cnn_norm_batch_var");
      this->core->template addArgument<ulong>(varId, normParamOffset);
      this->core->template addArgument<ulong>(varId, currentShape.c);
      this->core->template addArgument<ulong>(varId, currentShape.h);
      this->core->template addArgument<ulong>(varId, currentShape.w);
      this->core->template addArgument<ulong>(varId, static_cast<ulong>(1)); // N = 1
      this->core->template addArgument<ulong>(varId, static_cast<ulong>(0)); // sample_stride = 0
      this->core->template addArgument<ulong>(varId, inOffset); // actv_layer_offset = inOffset

      std::string normId = "calculate_norm_normalize_" + sStr + "_layer" + layerStr;
      this->core->addKernel(normId, "calculate_norm_normalize", size, 0);
      this->core->template addArgument<T>(normId, "cnn_actvs");
      this->core->template addArgument<T>(normId, "cnn_norm_xnorm");
      this->core->template addArgument<T>(normId, "cnn_norm_gamma");
      this->core->template addArgument<T>(normId, "cnn_norm_beta");
      this->core->template addArgument<T>(normId, "cnn_norm_batch_mean");
      this->core->template addArgument<T>(normId, "cnn_norm_batch_var");
      this->core->template addArgument<ulong>(normId, inOffset);
      this->core->template addArgument<ulong>(normId, outOffset);
      this->core->template addArgument<ulong>(normId, sampleIdx * sampleStride + layerActvOffset);
      this->core->template addArgument<ulong>(normId, normParamOffset);
      this->core->template addArgument<ulong>(normId, currentShape.c);
      this->core->template addArgument<ulong>(normId, currentShape.h);
      this->core->template addArgument<ulong>(normId, currentShape.w);
      this->core->template addArgument<float>(normId, norm.epsilon);

      normIdx++;
      break;
    }

    case LayerType::BATCHNORM: {
      // True BatchNorm: batch-wide stats (N×H×W per channel)
      const auto& norm = std::get<NormLayerConfig>(layerConfig.config);
      ulong size = currentShape.size();
      ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;
      ulong layerActvOffset = this->bufferManager.layerInfos[i].actvOffset;

      if (batchSize > 1) {
        // Training: compute batch-wide mean/var (only once, for sample 0)
        if (sampleIdx == 0) {
          ulong localWS = 256;
          ulong meanGlobalWS = currentShape.c * localWS;

          std::string meanId = "calculate_norm_mean_layer" + layerStr;
          this->core->addKernel(meanId, "calculate_norm_mean", meanGlobalWS, 0, localWS);
          this->core->template addArgument<T>(meanId, "cnn_actvs");
          this->core->template addArgument<T>(meanId, "cnn_norm_batch_mean");
          this->core->template addArgument<ulong>(meanId, normParamOffset);
          this->core->template addArgument<ulong>(meanId, currentShape.c);
          this->core->template addArgument<ulong>(meanId, currentShape.h);
          this->core->template addArgument<ulong>(meanId, currentShape.w);
          this->core->template addArgument<ulong>(meanId, batchSize);
          this->core->template addArgument<ulong>(meanId, sampleStride);
          this->core->template addArgument<ulong>(meanId, layerActvOffset);

          std::string varId = "calculate_norm_var_layer" + layerStr;
          this->core->addKernel(varId, "calculate_norm_var", meanGlobalWS, 0, localWS);
          this->core->template addArgument<T>(varId, "cnn_actvs");
          this->core->template addArgument<T>(varId, "cnn_norm_batch_mean");
          this->core->template addArgument<T>(varId, "cnn_norm_batch_var");
          this->core->template addArgument<ulong>(varId, normParamOffset);
          this->core->template addArgument<ulong>(varId, currentShape.c);
          this->core->template addArgument<ulong>(varId, currentShape.h);
          this->core->template addArgument<ulong>(varId, currentShape.w);
          this->core->template addArgument<ulong>(varId, batchSize);
          this->core->template addArgument<ulong>(varId, sampleStride);
          this->core->template addArgument<ulong>(varId, layerActvOffset);
        }

        // Per-sample normalize using batch-wide stats
        std::string normId = "calculate_norm_normalize_" + sStr + "_layer" + layerStr;
        this->core->addKernel(normId, "calculate_norm_normalize", size, 0);
        this->core->template addArgument<T>(normId, "cnn_actvs");
        this->core->template addArgument<T>(normId, "cnn_norm_xnorm");
        this->core->template addArgument<T>(normId, "cnn_norm_gamma");
        this->core->template addArgument<T>(normId, "cnn_norm_beta");
        this->core->template addArgument<T>(normId, "cnn_norm_batch_mean");
        this->core->template addArgument<T>(normId, "cnn_norm_batch_var");
        this->core->template addArgument<ulong>(normId, inOffset);
        this->core->template addArgument<ulong>(normId, outOffset);
        this->core->template addArgument<ulong>(normId, sampleIdx * sampleStride + layerActvOffset);
        this->core->template addArgument<ulong>(normId, normParamOffset);
        this->core->template addArgument<ulong>(normId, currentShape.c);
        this->core->template addArgument<ulong>(normId, currentShape.h);
        this->core->template addArgument<ulong>(normId, currentShape.w);
        this->core->template addArgument<float>(normId, norm.epsilon);
      } else {
        // Inference (batchSize == 1): use running stats
        std::string normId = "calculate_norm_normalize_" + sStr + "_layer" + layerStr;
        this->core->addKernel(normId, "calculate_norm_normalize", size, 0);
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
        this->core->template addArgument<float>(normId, norm.epsilon);
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
void GPUKernelBuilder<T>::addBackpropagateKernels(ulong sampleIdx)
{
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  ulong numLayers = cnnLayers.size();
  ulong batchSize = this->bufferManager.batchSize;

  ulong sampleStride = this->bufferManager.totalActvSize;
  ulong poolSampleStride = this->bufferManager.totalPoolIndexSize;
  std::string sStr = "s" + std::to_string(sampleIdx);

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

    case LayerType::BATCHNORM:
    case LayerType::INSTANCENORM:
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

    ulong gradInOffset = sampleIdx * sampleStride + this->bufferManager.layerInfos[static_cast<ulong>(i)].actvOffset;
    ulong gradOutOffset =
      sampleIdx * sampleStride + this->bufferManager.layerInfos[static_cast<ulong>(i) + 1].actvOffset;
    ulong actvInOffset = sampleIdx * sampleStride + this->bufferManager.layerInfos[static_cast<ulong>(i)].actvOffset;

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
      std::string filterId = "calculate_dCost_dFilters_" + sStr + "_layer" + layerStr;
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
      std::string biasId = "calculate_dCost_dBiases_" + sStr + "_layer" + layerStr;
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
        std::string inputId = "calculate_dCost_dInput_" + sStr + "_layer" + layerStr;
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
      std::string kernelId = "calculate_dCost_dRelu_" + sStr + "_layer" + layerStr;
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
      ulong poolIndexOffset = sampleIdx * poolSampleStride + this->bufferManager.poolInfos[poolIdx].indexOffset;

      ulong inSize = inShape.size();
      std::string zeroId = "zero_pool_grad_" + sStr + "_layer" + layerStr;
      this->core->addKernel(zeroId, "zero_buffer", inSize, 0);
      this->core->template addArgument<T>(zeroId, "cnn_grads");
      this->core->template addArgument<ulong>(zeroId, gradInOffset);
      this->core->template addArgument<ulong>(zeroId, inSize);

      ulong outSize = outShape.size();

      if (pool.poolType == PoolTypeEnum::MAX) {
        std::string poolId = "calculate_dCost_dMaxpool_" + sStr + "_layer" + layerStr;
        this->core->addKernel(poolId, "calculate_dCost_dMaxpool", outSize, 0);
        this->core->template addArgument<T>(poolId, "cnn_grads");
        this->core->template addArgument<ulong>(poolId, "cnn_pool_indices");
        this->core->template addArgument<ulong>(poolId, gradOutOffset);
        this->core->template addArgument<ulong>(poolId, poolIndexOffset);
        this->core->template addArgument<ulong>(poolId, outSize);
      } else {
        std::string poolId = "calculate_dCost_dAvgpool_" + sStr + "_layer" + layerStr;
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

    case LayerType::INSTANCENORM: {
      // InstanceNorm backward: per-image spatial stats (N=1)
      normIdx--;
      const auto& norm = std::get<NormLayerConfig>(layerConfig.config);
      ulong size = inShape.size();
      ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;

      ulong localWS = 256;
      ulong dgGlobalWS = inShape.c * localWS;
      std::string dgId = "calculate_norm_dGammaBeta_" + sStr + "_layer" + layerStr;
      this->core->addKernel(dgId, "calculate_norm_dGammaBeta", dgGlobalWS, 0, localWS);
      this->core->template addArgument<T>(dgId, "cnn_grads");
      this->core->template addArgument<T>(dgId, "cnn_norm_xnorm");
      this->core->template addArgument<T>(dgId, "cnn_norm_dGamma");
      this->core->template addArgument<T>(dgId, "cnn_norm_dBeta");
      this->core->template addArgument<ulong>(dgId, normParamOffset);
      this->core->template addArgument<ulong>(dgId, inShape.c);
      this->core->template addArgument<ulong>(dgId, inShape.h);
      this->core->template addArgument<ulong>(dgId, inShape.w);
      this->core->template addArgument<ulong>(dgId, static_cast<ulong>(1)); // N = 1
      this->core->template addArgument<ulong>(dgId, static_cast<ulong>(0)); // sample_stride = 0
      this->core->template addArgument<ulong>(dgId, gradOutOffset); // grad_layer_offset
      this->core->template addArgument<ulong>(dgId, actvInOffset); // xnorm_layer_offset

      ulong M = inShape.h * inShape.w; // InstanceNorm: M = H * W
      std::string diId = "calculate_norm_dInput_" + sStr + "_layer" + layerStr;
      this->core->addKernel(diId, "calculate_norm_dInput", size, 0);
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
      this->core->template addArgument<float>(diId, norm.epsilon);
      this->core->template addArgument<ulong>(diId, M);
      break;
    }

    case LayerType::BATCHNORM: {
      // True BatchNorm backward: batch-wide dGammaBeta, per-sample dInput
      normIdx--;
      const auto& norm = std::get<NormLayerConfig>(layerConfig.config);
      ulong size = inShape.size();
      ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;
      ulong layerActvOffset = this->bufferManager.layerInfos[static_cast<ulong>(i)].actvOffset;
      ulong layerGradOffset = this->bufferManager.layerInfos[static_cast<ulong>(i) + 1].actvOffset;

      // Batch-wide dGamma/dBeta (only once, for sample 0)
      if (sampleIdx == 0) {
        ulong localWS = 256;
        ulong dgGlobalWS = inShape.c * localWS;
        std::string dgId = "calculate_norm_dGammaBeta_layer" + layerStr;
        this->core->addKernel(dgId, "calculate_norm_dGammaBeta", dgGlobalWS, 0, localWS);
        this->core->template addArgument<T>(dgId, "cnn_grads");
        this->core->template addArgument<T>(dgId, "cnn_norm_xnorm");
        this->core->template addArgument<T>(dgId, "cnn_norm_dGamma");
        this->core->template addArgument<T>(dgId, "cnn_norm_dBeta");
        this->core->template addArgument<ulong>(dgId, normParamOffset);
        this->core->template addArgument<ulong>(dgId, inShape.c);
        this->core->template addArgument<ulong>(dgId, inShape.h);
        this->core->template addArgument<ulong>(dgId, inShape.w);
        this->core->template addArgument<ulong>(dgId, batchSize);
        this->core->template addArgument<ulong>(dgId, sampleStride);
        this->core->template addArgument<ulong>(dgId, layerGradOffset);
        this->core->template addArgument<ulong>(dgId, layerActvOffset);
      }

      // Per-sample dInput using batch-wide dGamma/dBeta and M = N * H * W
      ulong M = batchSize * inShape.h * inShape.w;
      std::string diId = "calculate_norm_dInput_" + sStr + "_layer" + layerStr;
      this->core->addKernel(diId, "calculate_norm_dInput", size, 0);
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
      this->core->template addArgument<float>(diId, norm.epsilon);
      this->core->template addArgument<ulong>(diId, M);
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
void GPUKernelBuilder<T>::addCopyBridgeKernels(ulong sampleIdx)
{
  ulong sampleStride = this->bufferManager.totalActvSize;
  ulong lastLayerIdx = this->bufferManager.layerInfos.size() - 1;
  ulong cnnOutputOffset = sampleIdx * sampleStride + this->bufferManager.layerInfos[lastLayerIdx].actvOffset;

  std::string kernelId = "copy_cnn_to_ann_s" + std::to_string(sampleIdx);
  this->core->addKernel(kernelId, "copy_cnn_to_ann", this->bufferManager.flattenSize, 0);
  this->core->template addArgument<T>(kernelId, "cnn_actvs");
  this->core->template addArgument<T>(kernelId, "actvs");
  this->core->template addArgument<ulong>(kernelId, cnnOutputOffset);
  this->core->template addArgument<ulong>(kernelId, this->bufferManager.flattenSize);
}

//===================================================================================================================//
//-- addCNNAccumulateKernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addCNNPerSampleAccumulateKernels(ulong sampleIdx)
{
  std::string sStr = "s" + std::to_string(sampleIdx);

  if (this->bufferManager.totalFilterSize > 0) {
    std::string id = "accumulate_gradients_filters_" + sStr;
    this->core->addKernel(id, "accumulate_gradients", this->bufferManager.totalFilterSize, 0);
    this->core->template addArgument<T>(id, "cnn_accum_dFilters");
    this->core->template addArgument<T>(id, "cnn_dFilters");
    this->core->template addArgument<ulong>(id, static_cast<ulong>(0));
    this->core->template addArgument<ulong>(id, this->bufferManager.totalFilterSize);
  }

  if (this->bufferManager.totalBiasSize > 0) {
    std::string id = "accumulate_gradients_biases_" + sStr;
    this->core->addKernel(id, "accumulate_gradients", this->bufferManager.totalBiasSize, 0);
    this->core->template addArgument<T>(id, "cnn_accum_dBiases");
    this->core->template addArgument<T>(id, "cnn_dBiases");
    this->core->template addArgument<ulong>(id, static_cast<ulong>(0));
    this->core->template addArgument<ulong>(id, this->bufferManager.totalBiasSize);
  }

  // InstanceNorm dGamma/dBeta are computed per-sample, so accumulate per-sample.
  // BatchNorm dGamma/dBeta are computed batch-wide (once), so only accumulate for sample 0.
  if (this->bufferManager.totalNormParamSize > 0) {
    bool hasInstanceNorm = false;
    bool hasBatchNorm = false;

    for (const auto& layer : this->coreConfig.layersConfig.cnnLayers) {
      if (layer.type == LayerType::INSTANCENORM)
        hasInstanceNorm = true;

      if (layer.type == LayerType::BATCHNORM)
        hasBatchNorm = true;
    }

    // Accumulate norm dGamma/dBeta if this sample has InstanceNorm contributions,
    // or if this is sample 0 and there are BatchNorm layers (batch-wide dGamma/dBeta computed once)
    if (hasInstanceNorm || (hasBatchNorm && sampleIdx == 0)) {
      std::string gammaId = "accumulate_gradients_norm_gamma_" + sStr;
      this->core->addKernel(gammaId, "accumulate_gradients", this->bufferManager.totalNormParamSize, 0);
      this->core->template addArgument<T>(gammaId, "cnn_accum_norm_dGamma");
      this->core->template addArgument<T>(gammaId, "cnn_norm_dGamma");
      this->core->template addArgument<ulong>(gammaId, static_cast<ulong>(0));
      this->core->template addArgument<ulong>(gammaId, this->bufferManager.totalNormParamSize);

      std::string betaId = "accumulate_gradients_norm_beta_" + sStr;
      this->core->addKernel(betaId, "accumulate_gradients", this->bufferManager.totalNormParamSize, 0);
      this->core->template addArgument<T>(betaId, "cnn_accum_norm_dBeta");
      this->core->template addArgument<T>(betaId, "cnn_norm_dBeta");
      this->core->template addArgument<ulong>(betaId, static_cast<ulong>(0));
      this->core->template addArgument<ulong>(betaId, this->bufferManager.totalNormParamSize);
    }
  }
}

//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addCNNNormStatsAccumulateKernels()
{
  if (this->bufferManager.totalNormParamSize > 0) {
    this->core->addKernel("accumulate_norm_mean", "accumulate_gradients", this->bufferManager.totalNormParamSize, 0);
    this->core->template addArgument<T>("accumulate_norm_mean", "cnn_norm_accum_mean");
    this->core->template addArgument<T>("accumulate_norm_mean", "cnn_norm_batch_mean");
    this->core->template addArgument<ulong>("accumulate_norm_mean", static_cast<ulong>(0));
    this->core->template addArgument<ulong>("accumulate_norm_mean", this->bufferManager.totalNormParamSize);

    this->core->addKernel("accumulate_norm_var", "accumulate_gradients", this->bufferManager.totalNormParamSize, 0);
    this->core->template addArgument<T>("accumulate_norm_var", "cnn_norm_accum_var");
    this->core->template addArgument<T>("accumulate_norm_var", "cnn_norm_batch_var");
    this->core->template addArgument<ulong>("accumulate_norm_var", static_cast<ulong>(0));
    this->core->template addArgument<ulong>("accumulate_norm_var", this->bufferManager.totalNormParamSize);
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
    // Get momentum from first norm layer config
    float momentum = 0.1f;

    for (const auto& layerConfig : this->coreConfig.layersConfig.cnnLayers) {
      if (layerConfig.type == LayerType::BATCHNORM || layerConfig.type == LayerType::INSTANCENORM) {
        const auto& norm = std::get<NormLayerConfig>(layerConfig.config);
        momentum = norm.momentum;
        break;
      }
    }

    for (ulong n = 0; n < this->bufferManager.normInfos.size(); n++) {
      ulong normParamOffset = this->bufferManager.normInfos[n].paramOffset;
      ulong numChannels = this->bufferManager.normInfos[n].numChannels;
      std::string kernelId = "update_norm_running_stats_" + std::to_string(n);
      this->core->addKernel(kernelId, "update_norm_running_stats", numChannels, 0);
      this->core->template addArgument<T>(kernelId, "cnn_norm_running_mean");
      this->core->template addArgument<T>(kernelId, "cnn_norm_running_var");
      this->core->template addArgument<T>(kernelId, "cnn_norm_accum_mean");
      this->core->template addArgument<T>(kernelId, "cnn_norm_accum_var");
      this->core->template addArgument<ulong>(kernelId, normParamOffset);
      this->core->template addArgument<ulong>(kernelId, numChannels);
      this->core->template addArgument<ulong>(kernelId, numSamples);
      this->core->template addArgument<float>(kernelId, momentum);
    }
  }
}

//===================================================================================================================//
// Explicit template instantiations.
//===================================================================================================================//

template class CNN::GPUKernelBuilder<int>;
template class CNN::GPUKernelBuilder<float>;
template class CNN::GPUKernelBuilder<double>;