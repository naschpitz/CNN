#include "CNN_SlidingStrategy.hpp"

#include <stdexcept>

using namespace CNN;

//===================================================================================================================//

SlidingStrategyType SlidingStrategy::nameToType(const std::string& name)
{
  auto it = slidingStrategyMap.find(name);

  if (it != slidingStrategyMap.end()) {
    return it->second;
  }

  throw std::runtime_error("Unknown sliding strategy: " + name);
}

//===================================================================================================================//

std::string SlidingStrategy::typeToName(const SlidingStrategyType& type)
{
  for (const auto& pair : slidingStrategyMap) {
    if (pair.second == type) {
      return pair.first;
    }
  }

  throw std::runtime_error("Unknown sliding strategy enum value");
}

//===================================================================================================================//

ulong SlidingStrategy::computePadding(ulong kernelSize, SlidingStrategyType strategy)
{
  switch (strategy) {
  case SlidingStrategyType::VALID:
    return 0;
  case SlidingStrategyType::FULL:
    return kernelSize - 1;
  case SlidingStrategyType::SAME:
    return kernelSize / 2;
  default:
    throw std::runtime_error("Cannot compute padding for unknown sliding strategy");
  }
}
