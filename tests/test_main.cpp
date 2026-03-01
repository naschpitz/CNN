#include "test_helpers.hpp"

int testsPassed = 0;
int testsFailed = 0;

void runConv2DTests();
void runLayerTests();
void runIntegrationTests();
void runGPUTests();

int main()
{
  std::cout << "=== CNN Unit Tests ===" << std::endl;
  runConv2DTests();
  runLayerTests();

  std::cout << std::endl;
  std::cout << "=== Integration Tests ===" << std::endl;
  runIntegrationTests();

  std::cout << std::endl;
  std::cout << "=== GPU Tests ===" << std::endl;
  runGPUTests();

  std::cout << std::endl;
  std::cout << "=== Results: " << testsPassed << " passed, " << testsFailed << " failed ===" << std::endl;
  return (testsFailed > 0) ? 1 : 0;
}
