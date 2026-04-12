#include "CNN_TrainingMonitor.hpp"

#include <cmath>
#include <limits>
#include <sstream>

using namespace CNN;

//===================================================================================================================//

template <typename T>
TrainingMonitor<T>::TrainingMonitor(const MonitoringConfig& config)
  : config(config),
    bestLoss_(std::numeric_limits<T>::max()),
    bestEpoch_(0),
    epochsWithoutImprovement(0),
    newBest_(false),
    stopReason_()
{
}

//===================================================================================================================//

template <typename T>
bool TrainingMonitor<T>::checkEpoch(ulong epoch, T epochLoss, std::optional<T> validationLoss)
{
  newBest_ = false;

  // Use validation loss when available, otherwise training loss
  T loss = validationLoss.has_value() ? validationLoss.value() : epochLoss;

  // Update best loss tracking
  T delta = static_cast<T>(config.metrics.lossStagnation.minDelta);

  if (loss < bestLoss_ - delta) {
    bestLoss_ = loss;
    bestEpoch_ = epoch;
    newBest_ = true;
    epochsWithoutImprovement = 0;
  }

  // Only perform metric checks at checkInterval boundaries
  if (epoch % config.checkInterval != 0) {
    return false;
  }

  // Check loss explosion (immediate stop)
  if (checkLossExplosion(loss)) {
    return true;
  }

  // Check loss stagnation (patience-based)
  if (!newBest_) {
    if (checkLossStagnation(loss)) {
      return true;
    }
  }

  return false;
}

//===================================================================================================================//

template <typename T>
bool TrainingMonitor<T>::isNewBest() const
{
  return newBest_;
}

//===================================================================================================================//

template <typename T>
std::string TrainingMonitor<T>::stopReason() const
{
  return stopReason_;
}

//===================================================================================================================//

template <typename T>
ulong TrainingMonitor<T>::bestEpoch() const
{
  return bestEpoch_;
}

//===================================================================================================================//

template <typename T>
T TrainingMonitor<T>::bestLoss() const
{
  return bestLoss_;
}

//===================================================================================================================//
//-- Metric evaluators --//
//===================================================================================================================//

template <typename T>
bool TrainingMonitor<T>::checkLossStagnation(T loss)
{
  if (!config.metrics.lossStagnation.enabled) {
    return false;
  }

  epochsWithoutImprovement++;

  if (epochsWithoutImprovement >= config.patience) {
    std::ostringstream oss;
    oss << "Loss stagnation: no improvement > " << config.metrics.lossStagnation.minDelta << " for " << config.patience
        << " check intervals (" << config.patience * config.checkInterval << " epochs)";
    stopReason_ = oss.str();
    return true;
  }

  return false;
}

//===================================================================================================================//

template <typename T>
bool TrainingMonitor<T>::checkLossExplosion(T loss)
{
  if (!config.metrics.lossExplosion.enabled) {
    return false;
  }

  if (std::isnan(static_cast<double>(loss))) {
    stopReason_ = "Loss explosion: NaN detected";
    return true;
  }

  // Only check threshold if we have a valid best loss
  if (bestLoss_ < std::numeric_limits<T>::max()) {
    T threshold = static_cast<T>(config.metrics.lossExplosion.threshold);

    if (loss > threshold * bestLoss_) {
      std::ostringstream oss;
      oss << "Loss explosion: current loss (" << loss << ") exceeds " << config.metrics.lossExplosion.threshold
          << "x best loss (" << bestLoss_ << ")";
      stopReason_ = oss.str();
      return true;
    }
  }

  return false;
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::TrainingMonitor<int>;
template class CNN::TrainingMonitor<double>;
template class CNN::TrainingMonitor<float>;
