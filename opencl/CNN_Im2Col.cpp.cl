#ifndef CNN_IM2COL_CPP_CL
#define CNN_IM2COL_CPP_CL

// Note: Depends on CNN_Defines.hpp.cl (TYPE)

//===================================================================================================================//

// im2col: rearrange input patches into a column matrix for GEMM-based convolution
// Input tensor: (C_in, H, W) at inputOffset in the input buffer
// Output matrix: (C_in * kH * kW, outH * outW) in the col buffer
// One work-item per output element in the column matrix
// nElements = C_in * kH * kW * outH * outW
kernel void im2col(global TYPE* input, global TYPE* col, ulong inputOffset, ulong C, ulong H, ulong W, ulong kH,
                   ulong kW, ulong strideY, ulong strideX, ulong padY, ulong padX, ulong outH, ulong outW)
{
  size_t gid = get_global_id(0);

  ulong colRows = C * kH * kW;
  ulong colCols = outH * outW;
  ulong totalElements = colRows * colCols;

  if (gid >= totalElements)
    return;

  // Decompose gid into (row, col_idx) in the column matrix
  ulong row = gid / colCols;
  ulong colIdx = gid % colCols;

  // Decompose row into (c, kh, kw) — which channel and filter position
  ulong c = row / (kH * kW);
  ulong rem = row % (kH * kW);
  ulong kh = rem / kW;
  ulong kw = rem % kW;

  // Decompose colIdx into (oh, ow) — which output position
  ulong oh = colIdx / outW;
  ulong ow = colIdx % outW;

  // Compute input position
  long ih = (long)(oh * strideY + kh) - (long)padY;
  long iw = (long)(ow * strideX + kw) - (long)padX;

  TYPE val = (TYPE)0;

  if (ih >= 0 && ih < (long)H && iw >= 0 && iw < (long)W)
    val = input[inputOffset + c * H * W + (ulong)ih * W + (ulong)iw];

  col[gid] = val;
}

//===================================================================================================================//

// col2im: scatter column matrix back to input tensor (inverse of im2col)
// One work-item per input element (c, ih, iw) — iterates over all output positions
// whose receptive field includes this input element and sums contributions.
// This avoids float atomics.
// The output gradient buffer must be zeroed before calling this kernel.
// nElements = C * H * W
kernel void col2im(global TYPE* col, global TYPE* output, ulong outputOffset, ulong C, ulong H, ulong W, ulong kH,
                   ulong kW, ulong strideY, ulong strideX, ulong padY, ulong padX, ulong outH, ulong outW)
{
  size_t gid = get_global_id(0);

  ulong totalElements = C * H * W;

  if (gid >= totalElements)
    return;

  // Decompose gid into (c, ih, iw)
  ulong c = gid / (H * W);
  ulong rem = gid % (H * W);
  ulong ih = rem / W;
  ulong iw = rem % W;

  TYPE sum = (TYPE)0;

  ulong colCols = outH * outW;

  // Iterate over all output positions (oh, ow) whose receptive field includes (ih, iw)
  // oh * strideY + kh - padY = ih  →  kh = ih + padY - oh * strideY
  // For kh to be in [0, kH), we need: oh * strideY <= ih + padY < oh * strideY + kH
  // So: oh >= (ih + padY - kH + 1) / strideY  and  oh <= (ih + padY) / strideY

  long ihPad = (long)ih + (long)padY;
  long iwPad = (long)iw + (long)padX;

  // Compute oh range
  long ohStart = (ihPad - (long)kH + 1 + (long)strideY - 1) / (long)strideY; // ceil division

  if (ohStart < 0)
    ohStart = 0;
  long ohEnd = ihPad / (long)strideY;

  if (ohEnd >= (long)outH)
    ohEnd = (long)outH - 1;

  // Compute ow range
  long owStart = (iwPad - (long)kW + 1 + (long)strideX - 1) / (long)strideX;

  if (owStart < 0)
    owStart = 0;
  long owEnd = iwPad / (long)strideX;

  if (owEnd >= (long)outW)
    owEnd = (long)outW - 1;

  for (long oh = ohStart; oh <= ohEnd; oh++) {
    long kh = ihPad - oh * (long)strideY;

    for (long ow = owStart; ow <= owEnd; ow++) {
      long kw = iwPad - ow * (long)strideX;

      // Column matrix row = c * kH * kW + kh * kW + kw
      // Column matrix col = oh * outW + ow
      ulong colRow = c * kH * kW + (ulong)kh * kW + (ulong)kw;
      ulong colCol = (ulong)oh * outW + (ulong)ow;

      sum += col[colRow * colCols + colCol];
    }
  }

  output[outputOffset + gid] = sum;
}

#endif // CNN_IM2COL_CPP_CL
