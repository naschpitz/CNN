#ifndef CNN_VALIDATIONCONFIG_HPP
#define CNN_VALIDATIONCONFIG_HPP

#include <sys/types.h>

//===================================================================================================================//

namespace CNN
{
  struct ValidationConfig {
    bool enabled = true;
    bool autoSize = true;
    float size = 0.15f;
    ulong checkInterval = 1;
  };
}

//===================================================================================================================//

#endif // CNN_VALIDATIONCONFIG_HPP
