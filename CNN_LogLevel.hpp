#ifndef CNN_LOGLEVEL_HPP
#define CNN_LOGLEVEL_HPP

//===================================================================================================================//

namespace CNN {
  enum class LogLevel : int {
    QUIET   = 0,
    ERROR   = 1,
    WARNING = 2,
    INFO    = 3,
    DEBUG   = 4
  };
}

//===================================================================================================================//

#endif // CNN_LOGLEVEL_HPP

