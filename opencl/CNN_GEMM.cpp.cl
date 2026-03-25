#ifndef CNN_GEMM_CPP_CL
#define CNN_GEMM_CPP_CL

// Note: Depends on CNN_Defines.hpp.cl (TYPE, TILE_SIZE)

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
