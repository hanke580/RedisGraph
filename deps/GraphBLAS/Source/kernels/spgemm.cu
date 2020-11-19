#include <stdio.h>

// Sparse matrix-Sparse matrix multiplication with sparse matrix mask
// Strategy:
// 1) Loop through mask using 1 warp/row
// 2) For each nonzero (row, col) of mask:
//    i)   initialize each thread to identity
//    ii)  compute dot-product A(row, :) x B(:, col)
//    iii) use warp on each nonzero at float time
//    iv)  tally up accumulated sum using warp reduction
//    v)   write to global memory C_csrVal
__global__ void spgemmMaskedKernel(float*           C_csrVal,
                                   const int* mask_csrRowPtr,
                                   const int* mask_csrColInd,
                                   float*           mask_csrVal,
                                   GxB_binary_function        mul_op,
                                   GxB_binary_function        add_op,
                                   float            identity,
                                   const int* A_csrRowPtr,
                                   const int* A_csrColInd,
                                   const float*     A_csrVal,
                                   const int* B_cscColPtr,
                                   const int* B_cscRowInd,
                                   const float*     B_cscVal,
                                   int        A_nrows) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id   = thread_id / 32;
  int lane_id   = thread_id & (32 - 1);
  if (warp_id < A_nrows) {
    int row_start = mask_csrRowPtr[warp_id];
    int row_end   = mask_csrRowPtr[warp_id+1];

    // Entire warp works together on each nonzero
    for (int edge = row_start; edge < row_end; ++edge) {
      float mask_val = mask_csrVal[edge];
      float accumulator = 0;
      if (mask_val) {
        // Load B bounds on which we must do binary search
        int B_ind       = mask_csrColInd[edge];
        int B_col_start = B_cscColPtr[B_ind];
        int B_col_end   = B_cscColPtr[B_ind+1];
        
        // Each thread iterates along row
        // Does binary search on B_row to try to find A_col
        // Adds result to accumulator if found
        int ind = row_start + lane_id;
        for (int ind_start = row_start; ind_start < row_end;
            ind_start += 32) {
          if (ind < row_end) {
            int A_col = A_csrColInd[ind];
            int B_row = binarySearch(B_cscRowInd, A_col, B_col_start,
                B_col_end);

            if (B_row != -1) {
              float A_t = A_csrVal[ind];
              float B_t = B_cscVal[B_row];
              float C_t = A_t * B_t;
              accumulator = C_t + accumulator;
            }
          }
          ind += 32;
        }

        // Warp reduce for each edge
        for (int i = 1; i < 32; i *= 2)
          accumulator = __shfl_xor_sync(-1, accumulator, i) + accumulator;
      }
      // Write to output
      if (lane_id == 0)
        C_csrVal[edge] = accumulator;
    }
  }
}