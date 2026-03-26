#ifndef CNN_GEMM_CPP_CL
#define CNN_GEMM_CPP_CL

// Tiled General Matrix Multiply (GEMM) kernels for im2col-based convolution.
//
// WHY GEMM?
// Direct convolution (one work-item per output element, nested loops over filter×channels)
// has poor memory access patterns: each work-item reads scattered, non-contiguous memory
// locations with no data reuse between neighboring work-items. This wastes GPU memory
// bandwidth, which is the primary bottleneck.
//
// The im2col + GEMM approach reshapes convolution into a matrix multiplication:
//   Output = Filters × im2col(Input) + Bias
// Matrix multiplication has regular, predictable access patterns that GPUs excel at.
// By tiling the computation into TILE_SIZE × TILE_SIZE blocks loaded into fast local
// (shared) memory, each element is loaded once from global memory and reused TILE_SIZE
// times, reducing global memory traffic by ~TILE_SIZE×.
//
// HOW IT WORKS:
// Each work-group computes one TILE×TILE block of the output matrix C.
// The K dimension (shared between A and B) is processed in tiles:
//   1. All work-items collaboratively load one tile of A and one tile of B into local memory
//   2. barrier() ensures the tile is fully loaded
//   3. Each work-item computes a partial dot product using the local tiles
//   4. barrier() ensures the tile is fully consumed before loading the next one
//   5. After all K-tiles, the accumulated result is written to global memory
//
// Three variants handle the different matrix layouts needed for forward and backward passes:
//   gemm       — C = A × B + bias  (forward: Output = Filters × im2col + Bias)
//   gemm_transA — C = A^T × B      (backward: dInput_cols = Filters^T × dOut)
//   gemm_transB — C = A × B^T      (backward: dFilters = dOut × im2col^T)
//
// Depends on CNN_Defines.hpp.cl for TYPE and TILE_SIZE.

//===================================================================================================================//

// Tiled GEMM: C = A × B + bias
// A: (M, K) at offsetA, B: (K, N) at offsetB, C: (M, N) at offsetC
// bias: vector of length M at offsetBias, added per-row
// 2D dispatch: global (ceil(N/TILE_SIZE)*TILE_SIZE, ceil(M/TILE_SIZE)*TILE_SIZE)
//              local  (TILE_SIZE, TILE_SIZE)
kernel void gemm(global TYPE* A, global TYPE* B, global TYPE* C, global TYPE* bias, ulong offsetA, ulong offsetB,
                 ulong offsetC, ulong offsetBias, ulong M, ulong N, ulong K)
{
  uint localRow = get_local_id(1);
  uint localCol = get_local_id(0);
  uint globalRow = get_group_id(1) * TILE_SIZE + localRow;
  uint globalCol = get_group_id(0) * TILE_SIZE + localCol;

  local TYPE tileA[TILE_SIZE][TILE_SIZE];
  local TYPE tileB[TILE_SIZE][TILE_SIZE];

  TYPE acc = (TYPE)0;

  uint numTilesK = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (uint t = 0; t < numTilesK; t++) {
    uint aCol = t * TILE_SIZE + localCol;
    uint bRow = t * TILE_SIZE + localRow;

    // Load A[globalRow][aCol]
    if (globalRow < M && aCol < K)
      tileA[localRow][localCol] = A[offsetA + globalRow * K + aCol];
    else
      tileA[localRow][localCol] = (TYPE)0;

    // Load B[bRow][globalCol]
    if (bRow < K && globalCol < N)
      tileB[localRow][localCol] = B[offsetB + bRow * N + globalCol];
    else
      tileB[localRow][localCol] = (TYPE)0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < TILE_SIZE; k++)
      acc += tileA[localRow][k] * tileB[k][localCol];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (globalRow < M && globalCol < N)
    C[offsetC + globalRow * N + globalCol] = acc + bias[offsetBias + globalRow];
}

//===================================================================================================================//

// Tiled GEMM: C = A^T × B (no bias)
// A stored as (K, M), accessed as (M, K) via transposed indexing
// B: (K, N), C: (M, N)
kernel void gemm_transA(global TYPE* A, global TYPE* B, global TYPE* C, ulong offsetA, ulong offsetB, ulong offsetC,
                        ulong M, ulong N, ulong K)
{
  uint localRow = get_local_id(1);
  uint localCol = get_local_id(0);
  uint globalRow = get_group_id(1) * TILE_SIZE + localRow;
  uint globalCol = get_group_id(0) * TILE_SIZE + localCol;

  local TYPE tileA[TILE_SIZE][TILE_SIZE];
  local TYPE tileB[TILE_SIZE][TILE_SIZE];

  TYPE acc = (TYPE)0;

  uint numTilesK = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (uint t = 0; t < numTilesK; t++) {
    uint aCol = t * TILE_SIZE + localCol;
    uint bRow = t * TILE_SIZE + localRow;

    // Load A^T[globalRow][aCol] = A[aCol][globalRow]
    if (globalRow < M && aCol < K)
      tileA[localRow][localCol] = A[offsetA + aCol * M + globalRow];
    else
      tileA[localRow][localCol] = (TYPE)0;

    // Load B[bRow][globalCol]
    if (bRow < K && globalCol < N)
      tileB[localRow][localCol] = B[offsetB + bRow * N + globalCol];
    else
      tileB[localRow][localCol] = (TYPE)0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < TILE_SIZE; k++)
      acc += tileA[localRow][k] * tileB[k][localCol];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (globalRow < M && globalCol < N)
    C[offsetC + globalRow * N + globalCol] = acc;
}

//===================================================================================================================//

// Tiled GEMM: C = A × B^T (no bias)
// A: (M, K), B stored as (N, K), accessed as (K, N) via transposed indexing
// C: (M, N)
kernel void gemm_transB(global TYPE* A, global TYPE* B, global TYPE* C, ulong offsetA, ulong offsetB, ulong offsetC,
                        ulong M, ulong N, ulong K)
{
  uint localRow = get_local_id(1);
  uint localCol = get_local_id(0);
  uint globalRow = get_group_id(1) * TILE_SIZE + localRow;
  uint globalCol = get_group_id(0) * TILE_SIZE + localCol;

  local TYPE tileA[TILE_SIZE][TILE_SIZE];
  local TYPE tileB[TILE_SIZE][TILE_SIZE];

  TYPE acc = (TYPE)0;

  uint numTilesK = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (uint t = 0; t < numTilesK; t++) {
    uint aCol = t * TILE_SIZE + localCol;
    uint bRow = t * TILE_SIZE + localRow;

    // Load A[globalRow][aCol]
    if (globalRow < M && aCol < K)
      tileA[localRow][localCol] = A[offsetA + globalRow * K + aCol];
    else
      tileA[localRow][localCol] = (TYPE)0;

    // Load B^T[bRow][globalCol] = B[globalCol][bRow]
    if (bRow < K && globalCol < N)
      tileB[localRow][localCol] = B[offsetB + globalCol * K + bRow];
    else
      tileB[localRow][localCol] = (TYPE)0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < TILE_SIZE; k++)
      acc += tileA[localRow][k] * tileB[k][localCol];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (globalRow < M && globalCol < N)
    C[offsetC + globalRow * N + globalCol] = acc;
}

#endif // CNN_GEMM_CPP_CL
