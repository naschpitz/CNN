#ifndef CNN_GEMM_CPP_CL
#define CNN_GEMM_CPP_CL

// Register-blocked General Matrix Multiply (GEMM) kernels for im2col-based convolution.
//
// Each work-group is TILE_SIZE × TILE_SIZE threads (16×16 = 256 threads).
// Each thread computes a WPT × WPT sub-block of the output (4×4 = 16 elements).
// So each work-group computes a (TILE_SIZE*WPT) × (TILE_SIZE*WPT) output tile (64×64).
//
// The key optimization is register blocking: shared memory tiles are TILE_SIZE × TILE_SIZE,
// but each thread accumulates WPT×WPT results in private registers. For each K-tile step,
// each thread loads WPT elements from the A-tile column and WPT elements from the B-tile row,
// then performs WPT×WPT = 16 FMAs. This gives WPT² FMAs per 2×WPT loads = 2× better
// arithmetic intensity than the naive 1-element-per-thread approach.
//
// Three variants handle the different matrix layouts needed for forward and backward passes:
//   gemm       — C = A × B + bias  (forward: Output = Filters × im2col + Bias)
//   gemm_transA — C = A^T × B      (backward: dInput_cols = Filters^T × dOut)
//   gemm_transB — C = A × B^T      (backward: dFilters = dOut × im2col^T)
//
// Depends on CNN_Defines.hpp.cl for TYPE, TILE_SIZE, and WPT.
//
// 2D dispatch for all variants:
//   global: (ceil(N / (TILE_SIZE*WPT)) * TILE_SIZE,  ceil(M / (TILE_SIZE*WPT)) * TILE_SIZE)
//   local:  (TILE_SIZE, TILE_SIZE)

//===================================================================================================================//

// Register-blocked GEMM: C = A × B + bias
// A: (M, K) at offsetA, B: (K, N) at offsetB, C: (M, N) at offsetC
// bias: vector of length M at offsetBias, added per-row
kernel void gemm(global TYPE* A, global TYPE* B, global TYPE* C, global TYPE* bias, ulong offsetA, ulong offsetB,
                 ulong offsetC, ulong offsetBias, ulong M, ulong N, ulong K)
{
  uint lr = get_local_id(1);
  uint lc = get_local_id(0);
  uint groupRow = get_group_id(1) * TILE_SIZE * WPT;
  uint groupCol = get_group_id(0) * TILE_SIZE * WPT;

  local TYPE tileA[TILE_SIZE * WPT][TILE_SIZE];
  local TYPE tileB[TILE_SIZE][TILE_SIZE * WPT];

  TYPE acc[WPT][WPT];

  for (uint wr = 0; wr < WPT; wr++)

    for (uint wc = 0; wc < WPT; wc++)
      acc[wr][wc] = (TYPE)0;

  uint numTilesK = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (uint t = 0; t < numTilesK; t++) {
    uint tileK = t * TILE_SIZE;

    // Collaboratively load TILE_SIZE*WPT rows × TILE_SIZE cols of A
    for (uint wr = 0; wr < WPT; wr++) {
      uint aRow = groupRow + lr * WPT + wr;
      uint aCol = tileK + lc;

      if (aRow < M && aCol < K)
        tileA[lr * WPT + wr][lc] = A[offsetA + aRow * K + aCol];
      else
        tileA[lr * WPT + wr][lc] = (TYPE)0;
    }

    // Collaboratively load TILE_SIZE rows × TILE_SIZE*WPT cols of B
    for (uint wc = 0; wc < WPT; wc++) {
      uint bRow = tileK + lr;
      uint bCol = groupCol + lc * WPT + wc;

      if (bRow < K && bCol < N)
        tileB[lr][lc * WPT + wc] = B[offsetB + bRow * N + bCol];
      else
        tileB[lr][lc * WPT + wc] = (TYPE)0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < TILE_SIZE; k++) {
      for (uint wr = 0; wr < WPT; wr++) {
        TYPE aVal = tileA[lr * WPT + wr][k];

        for (uint wc = 0; wc < WPT; wc++)
          acc[wr][wc] += aVal * tileB[k][lc * WPT + wc];
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (uint wr = 0; wr < WPT; wr++) {
    uint row = groupRow + lr * WPT + wr;

    for (uint wc = 0; wc < WPT; wc++) {
      uint col = groupCol + lc * WPT + wc;

      if (row < M && col < N)
        C[offsetC + row * N + col] = acc[wr][wc] + bias[offsetBias + row];
    }
  }
}

//===================================================================================================================//

// Register-blocked GEMM: C = A^T × B (no bias)
// A stored as (K, M), accessed as (M, K) via transposed indexing
// B: (K, N), C: (M, N)
kernel void gemm_transA(global TYPE* A, global TYPE* B, global TYPE* C, ulong offsetA, ulong offsetB, ulong offsetC,
                        ulong M, ulong N, ulong K)
{
  uint lr = get_local_id(1);
  uint lc = get_local_id(0);
  uint groupRow = get_group_id(1) * TILE_SIZE * WPT;
  uint groupCol = get_group_id(0) * TILE_SIZE * WPT;

  local TYPE tileA[TILE_SIZE * WPT][TILE_SIZE];
  local TYPE tileB[TILE_SIZE][TILE_SIZE * WPT];

  TYPE acc[WPT][WPT];

  for (uint wr = 0; wr < WPT; wr++)

    for (uint wc = 0; wc < WPT; wc++)
      acc[wr][wc] = (TYPE)0;

  uint numTilesK = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (uint t = 0; t < numTilesK; t++) {
    uint tileK = t * TILE_SIZE;

    // Load A^T: tileA[lr*WPT+wr][lc] = A^T[groupRow + lr*WPT + wr][tileK + lc] = A[tileK + lc][groupRow + lr*WPT + wr]
    for (uint wr = 0; wr < WPT; wr++) {
      uint aRow = groupRow + lr * WPT + wr;
      uint aCol = tileK + lc;

      if (aRow < M && aCol < K)
        tileA[lr * WPT + wr][lc] = A[offsetA + aCol * M + aRow];
      else
        tileA[lr * WPT + wr][lc] = (TYPE)0;
    }

    // Load B
    for (uint wc = 0; wc < WPT; wc++) {
      uint bRow = tileK + lr;
      uint bCol = groupCol + lc * WPT + wc;

      if (bRow < K && bCol < N)
        tileB[lr][lc * WPT + wc] = B[offsetB + bRow * N + bCol];
      else
        tileB[lr][lc * WPT + wc] = (TYPE)0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < TILE_SIZE; k++) {
      for (uint wr = 0; wr < WPT; wr++) {
        TYPE aVal = tileA[lr * WPT + wr][k];

        for (uint wc = 0; wc < WPT; wc++)
          acc[wr][wc] += aVal * tileB[k][lc * WPT + wc];
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (uint wr = 0; wr < WPT; wr++) {
    uint row = groupRow + lr * WPT + wr;

    for (uint wc = 0; wc < WPT; wc++) {
      uint col = groupCol + lc * WPT + wc;

      if (row < M && col < N)
        C[offsetC + row * N + col] = acc[wr][wc];
    }
  }
}

//===================================================================================================================//

// Register-blocked GEMM: C = A × B^T (no bias)
// A: (M, K), B stored as (N, K), accessed as (K, N) via transposed indexing
// C: (M, N)
kernel void gemm_transB(global TYPE* A, global TYPE* B, global TYPE* C, ulong offsetA, ulong offsetB, ulong offsetC,
                        ulong M, ulong N, ulong K)
{
  uint lr = get_local_id(1);
  uint lc = get_local_id(0);
  uint groupRow = get_group_id(1) * TILE_SIZE * WPT;
  uint groupCol = get_group_id(0) * TILE_SIZE * WPT;

  local TYPE tileA[TILE_SIZE * WPT][TILE_SIZE];
  local TYPE tileB[TILE_SIZE][TILE_SIZE * WPT];

  TYPE acc[WPT][WPT];

  for (uint wr = 0; wr < WPT; wr++)

    for (uint wc = 0; wc < WPT; wc++)
      acc[wr][wc] = (TYPE)0;

  uint numTilesK = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (uint t = 0; t < numTilesK; t++) {
    uint tileK = t * TILE_SIZE;

    // Load A
    for (uint wr = 0; wr < WPT; wr++) {
      uint aRow = groupRow + lr * WPT + wr;
      uint aCol = tileK + lc;

      if (aRow < M && aCol < K)
        tileA[lr * WPT + wr][lc] = A[offsetA + aRow * K + aCol];
      else
        tileA[lr * WPT + wr][lc] = (TYPE)0;
    }

    // Load B^T: tileB[lr][lc*WPT+wc] = B^T[tileK + lr][groupCol + lc*WPT + wc] = B[groupCol + lc*WPT + wc][tileK + lr]
    for (uint wc = 0; wc < WPT; wc++) {
      uint bRow = tileK + lr;
      uint bCol = groupCol + lc * WPT + wc;

      if (bRow < K && bCol < N)
        tileB[lr][lc * WPT + wc] = B[offsetB + bCol * K + bRow];
      else
        tileB[lr][lc * WPT + wc] = (TYPE)0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < TILE_SIZE; k++) {
      for (uint wr = 0; wr < WPT; wr++) {
        TYPE aVal = tileA[lr * WPT + wr][k];

        for (uint wc = 0; wc < WPT; wc++)
          acc[wr][wc] += aVal * tileB[k][lc * WPT + wc];
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (uint wr = 0; wr < WPT; wr++) {
    uint row = groupRow + lr * WPT + wr;

    for (uint wc = 0; wc < WPT; wc++) {
      uint col = groupCol + lc * WPT + wc;

      if (row < M && col < N)
        C[offsetC + row * N + col] = acc[wr][wc];
    }
  }
}

#endif // CNN_GEMM_CPP_CL
