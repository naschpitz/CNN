#ifndef CNN_DEVICE_HPP
#define CNN_DEVICE_HPP

#include <string>
#include <unordered_map>

//===================================================================================================================//

namespace CNN {
  enum class DeviceType {
    CPU,
    GPU,
    UNKNOWN
  };

  const std::unordered_map<std::string, DeviceType> deviceTypeMap = {
    {"cpu", DeviceType::CPU},
    {"gpu", DeviceType::GPU},
  };

  class Device
  {
    public:
      static DeviceType nameToType(const std::string& name);
      static std::string typeToName(const DeviceType& deviceType);
  };
}

//===================================================================================================================//

#endif // CNN_DEVICE_HPP

