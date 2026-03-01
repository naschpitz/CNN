#include "CNN_Device.hpp"

#include <stdexcept>

using namespace CNN;

//===================================================================================================================//

DeviceType Device::nameToType(const std::string& name)
{
  auto it = deviceTypeMap.find(name);

  if (it != deviceTypeMap.end()) {
    return it->second;
  }

  throw std::runtime_error("Unknown device type: " + name);
}

//===================================================================================================================//

std::string Device::typeToName(const DeviceType& deviceType)
{
  for (const auto& pair : deviceTypeMap) {
    if (pair.second == deviceType) {
      return pair.first;
    }
  }

  throw std::runtime_error("Unknown device type enum value");
}
