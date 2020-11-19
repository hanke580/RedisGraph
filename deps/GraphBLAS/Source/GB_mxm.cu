#include "kernels/spgemm.cu"
#include "GB_transpose.h"
#include "GB.h"

typedef long long  int64_t
typedef int TYPE

extern "C" GrB_Info GB_mxm_gpu
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_replace,           // if true, clear C before writing to it
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Semiring semiring,    // defines '+' and '*' for C=A*B
    const GrB_Matrix A,             // input matrix
    const bool A_transpose,         // if true, use A' instead of A
    const GrB_Matrix B,             // input matrix
    const bool B_transpose,         // if true, use B' instead of B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    const GrB_Desc_Value AxB_method,// for auto vs user selection of methods
    GB_Context Context
)

{
    if (A_transpose) {
        GB_OK(GB_tranpose(NULL, NULL, false, A, NULL, Context)); // 是否inplace
    } else if (A->is_csc == true) {
        printf("A csc not implemented yet");
        return;
        // csc => csr 参考 csc2csr的实现
    }
    if (B_transpose) {
        GB_OK(GB_tranpose(NULL, NULL, true, B, NULL, Context)); // 是否inplace
    } else if (B->is_csc == false) {
        printf("B csr not implemented yet");
        return;
        // csr => csc
    }

    // Check type A and B, and broadcast. For simplicity, choose A type currently.
    // GrB_Type A_Type = A->type;
    TYPE* C_csrVal;
    CUDA_CALL(cudaMemcpy(C_csrVal, M->x, M->nzmax * sizeof(TYPE),
    cudaMemcpyHostToDevice));

    int64_t* A_csrRowPtr;   // GPU CSR format
    int64_t* A_csrColInd;
    TYPE* A_csrVal;        // TODO: Need Correct A TYPE
    int64_t A_nrows = A->plen;

    // Alloc space in GPU and transfer memory from CPU to GPU
    CUDA_CALL(cudaMemcpy(A_csrRowPtr, A->p, (A->plen + 1) * sizeof(int64_t),
        cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(A_csrColInd, A->i, A->nzmax * sizeof(int64_t),
        cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(A_csrVal,    A->x, A->nzmax * sizeof(TYPE),
        cudaMemcpyHostToDevice));

    int64_t* B_cscColPtr;   // GPU CSR format
    int64_t* B_cscRowInd;
    TYPE* B_cscVal;        // TODO: Need Correct A TYPE

    // Alloc space in GPU and transfer memory from CPU to GPU
    CUDA_CALL(cudaMemcpy(B_cscColPtr, B->p, (B->plen + 1) * sizeof(int64_t),
        cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(B_cscRowInd, B->i, B->nzmax * sizeof(int64_t),
        cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(B_cscVal,    B->x, B->nzmax * sizeof(TYPE),
        cudaMemcpyHostToDevice));

    int64_t* M_csrRowPtr;       // GPU CSR format
    int64_t* M_csrColInd;
    TYPE* M_csrVal;             // TODO: Need Correct A TYPE

    CUDA_CALL(cudaMemcpy(M_csrRowPtr, M->p, (M->plen + 1) * sizeof(int64_t),
        cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(M_csrColInd, M->i, M->nzmax * sizeof(int64_t),
        cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(M_csrVal,    M->x, M->nzmax * sizeof(TYPE),
        cudaMemcpyHostToDevice));

    const int nt = 128;  // GrB_128
    if (Mask_comp) {
        if (C != A && C != B)
        CHECK(C->dup(&mask->sparse_));
        const SparseMatrix<m>* sparse_mask = &mask->sparse_;
        // Simple warp-per-row algorithm
        dim3 NT, NB;
        NT.x = nt;
        NT.y = 1;
        NT.z = 1;
        NB.x = (A_nrows + nt - 1) / nt * 32;
        NB.y = 1;
        NB.z = 1;

        spgemmMaskedKernel<<<NB, NT>>>(C->d_csrVal_,
            M_csrRowPtr,
            M_csrColInd,
            M_csrVal,
            semiring->multiply->function, semiring->add->op->function,
            *(semiring->add->identity),
            A_csrRowPtr, A_csrColInd, A_csrVal,
            B_cscColPtr, B_cscRowInd, B_cscVal,
            A_nrows);

        // Transfer data back
        CUDA_CALL(cudaMemcpy(C->x, C_csrVal, M->nzmax * sizeof(TYPE),
            cudaMemcpyHostToDevice));
    }
    else {
        printf("No Mask computing not complement\n");
    }
}