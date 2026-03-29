#include "test_helpers.hpp"

int testsPassed = 0;
int testsFailed = 0;

void runConv2DTests();
void runLayerTests();
void runLayerTests2();
void runInstanceNormTests();
void runGlobalAvgPoolTests();
void runGlobalDualPoolTests();
void runResidualTests();
void runIntegrationBasicTests();
void runIntegrationBasicTests2();
void runIntegrationBasicTests3();
void runIntegrationGlobalPoolTests();
void runIntegrationResidualTests();
void runIntegrationCostFuncTests();
void runIntegrationExactTests();
void runIntegrationBatchNormTests();
void runGPUBasicTests();
void runGPUBasicTests2();
void runGPUBasicTests3();
void runGPUBasicTests4();
void runGPUGlobalPoolTests();
void runGPUMultiGPUTests();
void runGPUMultiGPUTests2();
void runGPUExactTests();
void runGPUExactGlobalPoolTests();
void runGPUExactBNTests();
void runGPUBatchNormTests();
void runGPUBatchNormTests2();

int main()
{
  std::cout << "=== CNN Unit Tests ===" << std::endl;
  runConv2DTests();
  runLayerTests();
  runLayerTests2();
  runInstanceNormTests();
  runGlobalAvgPoolTests();
  runGlobalDualPoolTests();
  runResidualTests();

  std::cout << std::endl;
  std::cout << "=== Integration Tests ===" << std::endl;
  runIntegrationBasicTests();
  runIntegrationBasicTests2();
  runIntegrationBasicTests3();
  runIntegrationGlobalPoolTests();
  runIntegrationResidualTests();
  runIntegrationCostFuncTests();
  runIntegrationExactTests();
  runIntegrationBatchNormTests();

  std::cout << std::endl;
  std::cout << "=== GPU Tests ===" << std::endl;
  runGPUBasicTests();
  runGPUBasicTests2();
  runGPUBasicTests3();
  runGPUBasicTests4();
  runGPUGlobalPoolTests();
  runGPUMultiGPUTests();
  runGPUMultiGPUTests2();
  runGPUExactTests();
  runGPUExactGlobalPoolTests();
  runGPUExactBNTests();
  runGPUBatchNormTests();
  runGPUBatchNormTests2();

  std::cout << std::endl;
  std::cout << "=== Results: " << testsPassed << " passed, " << testsFailed << " failed ===" << std::endl;
  return (testsFailed > 0) ? 1 : 0;
}
