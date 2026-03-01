#include "CNN_PoolType.hpp"

#include <stdexcept>

using namespace CNN;

//===================================================================================================================//

PoolTypeEnum PoolType::nameToType(const std::string& name)
{
  auto it = poolTypeMap.find(name);

  if (it != poolTypeMap.end()) {
    return it->second;
  }

  throw std::runtime_error("Unknown pool type: " + name);
}

//===================================================================================================================//

std::string PoolType::typeToName(const PoolTypeEnum& type)
{
  for (const auto& pair : poolTypeMap) {
    if (pair.second == type) {
      return pair.first;
    }
  }

  throw std::runtime_error("Unknown pool type enum value");
}
