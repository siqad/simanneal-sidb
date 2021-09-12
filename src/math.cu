// @file:     math.cu
// @author:   Samuel
// @created:  2021.08.19
// @license:  Apache License 2.0
//
// @desc:     Implementation of general matrix multiply operations.

#include <cuda_runtime.h>

/**
 * Column major 0 index for arrays representing 2D matrices.
 */
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

/**
 * Add the two vectors.
 * @param v1 pointer to vector 1.
 * @param v2 pointer to vector 2.
 * @param out pointer to output vector.
 * @param dim size of the vectoris.
 */
template<typename T>
__device__ void vvAdd(T *v1, T *v2, T *out, int dim)
{
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i=t_id; i<dim; i+=stride) {
    out[i] = v1[i] + v2[i];
  }
}

/**
 * Add the two vectors with scaling factors applied.
 * @param v1 pointer to vector 1.
 * @param v2 pointer to vector 2.
 * @param alpha scaling factor to vector 1.
 * @param beta scaling factor to vector 2.
 * @param out pointer to output vector.
 * @param dim size of the vectoris.
 */
template<typename T>
__device__ void vvScaledAdd(T *v1, T *v2, T alpha, T beta, T *out, int dim)
{
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i=t_id; i<dim; i+=stride) {
    out[i] = alpha * v1[i] + beta * v2[i];
  }
}

/**
 * Compute a 2D matrix-matrix product.
 * @param m1 pointer to matrix 1.
 * @param m2 pointer to matrix 2.
 * @param out pointer to output matrix to write to (assumes mem already allocated).
 * @param r_dim_1 row dimension of m1.
 * @param c_dim_1 column dimension of m1.
 */
// TODO: explore allowing different types for m1 and m2. Need to think about how to pick type for out.
template<typename T>
__device__ void mmProd2D(T *m1, T *m2, T *out, int r_dim_1, int c_dim_1)
{
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int r1=t_id; r1<r_dim_1; r1+=stride) {
    for (int c1=0; c1<c_dim_1; c1++) {
      printf("r1=%d, c1=%d\n", r1, c1);
      out[IDX2C(r1, c1, c_dim_1)] = 0;
      for (int c2=0; c2<c_dim_1; c2++) {
        printf("out[%d]+=m1[%d]*m2[%d]\n", IDX2C(r1, c1, c_dim_1), IDX2C(r1, c2, c_dim_1), IDX2C(c2, r1, r_dim_1));
        out[IDX2C(r1, c1, c_dim_1)] += m1[IDX2C(r1, c2, c_dim_1)] * m2[IDX2C(c2, c1, r_dim_1)];
      }
    }
  }
}

/**
 * Compute a matrix-vector product.
 * @param m pointer to the matrix.
 * @param v pointer to the vector.
 * @param out pointer to output vector to write to (assumes mem already allocated).
 * @param r_dim_1 row dimension of m.
 * @param c_dim_1 column dimension of m.
 */
template<typename T>
__device__ void mvProd(T *m, T *v, T *out, int r_dim, int c_dim)
{
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int r=t_id; r<r_dim; r+=stride) {
    out[r] = 0;
    for (int c=0; c<c_dim; c++) {
      out[r] += m[IDX2C(r, c, c_dim)] * v[c];
    }
  }
}

/**
 * Extract a specific row/column from two matrices and perform an inner vector product on them.
 * @param m1 pointer to matrix 1.
 * @param m2 pointer to matrix 2.
 * @param out pointer to output var to write to.
 * @param arr_ind index of the array (row for m1, col for m2) to extract for inner-product.
 * @param r_dim_1 row dimension of m1.
 * @param c_dim_1 column dimension of m1.
 * @param temp_arr a temporary array pointer to use for temp storage, must be preallocated with size sizeof(T)*r_dim_1.
 */
template<typename T>
__device__ void vvInnerProdOfMatrixExtractedArrays(T *m1, T *m2, T *out, int arr_ind, int r_dim_1, int c_dim_1, T *temp_arr)
{
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  if (t_id == 0) {
    *out = 0;
  }
  __syncthreads();

  for (int n=t_id; n<r_dim_1; n+=stride) {
    //printf("%f * %f\n", m1[IDX2C(arr_ind, n, c_dim_1)],m2[IDX2C(n, arr_ind, c_dim_1)]);
    temp_arr[n] = m1[IDX2C(arr_ind, n, c_dim_1)] * m2[IDX2C(n, arr_ind, c_dim_1)];
    //printf("%f\n", temp_arr[n]);
  }
  __syncthreads();

  if (t_id == 0) {
    for (int n=0; n<r_dim_1; n++) {
      *out += temp_arr[n];
    }
  }
}

/**
 * Compute an inner vector product.
 * @param v1 pointer to vector 1.
 * @param v2 pointer to vector 2.
 * @param out pointer to output var to write to.
 * @param dim vector size.
 * @param temp_arr a temporary array point to use for temp storage, must be preallocated with size sizeof(T)*dim.
 */
template<typename T>
__device__ void vvInnerProd(T *v1, T *v2, T *out, int dim, T *temp_arr)
{
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  if (t_id == 0) {
    *out = 0;
  }
  __syncthreads();

  for (int n=t_id; n<dim; n+=stride) {
    temp_arr[n] = v1[n] * v2[n];
  }
  __syncthreads();

  if (t_id == 0) {
    for (int n=0; n<dim; n++) {
      *out += temp_arr[n];
    }
  }
}