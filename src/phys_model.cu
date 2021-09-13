// @file:     phys_model.cu
// @author:   Samuel
// @created:  2021.08.19
// @license:  Apache License 2.0
//
// @desc:     Implementation of some physical model functions.

#include <cuda_runtime.h>
#include "constants.h"
#include "math.cu"

/**
 * Calculate the system energy of the provided configuration.
 * @param n the charge configuration.
 * @param n_dbs count of DBs.
 * @param v_ij precomputed inter-DB repulsion.
 * @param v_ext external potentials at each DB site.
 * @param out output array to write to, memory must be preallocated.
 * @param temp_scalar temporary scalar preallocated to sizeof(TFloat).
 * @param temp_vec_0 temporary vector preallocated to n_dbs * sizeof(TFloat).
 * @param temp_vec_1 temporary vector preallocated to n_dbs * sizeof(TFloat).
 */
__device__ void calcSystemEnergy(float *n, int n_dbs, float *v_ij, float *v_ext, float *out, float *temp_scalar, float *temp_vec_0, float *temp_vec_1)
{
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float sum;
  __syncthreads();
  vvInnerProd(n, v_ext, temp_scalar, n_dbs, temp_vec_0);
  __syncthreads();
  if (t_id == 0) {
    sum += *temp_scalar;
    //printf("E_ext = %.3e\n", sum);
  }
  mvProd(v_ij, n, temp_vec_0, n_dbs, n_dbs);
  __syncthreads();
  //if (t_id == 0) {
  //  printf("Vij n = %.3e\n", *temp);
  //}
  vvInnerProd(n, temp_vec_0, temp_scalar, n_dbs, temp_vec_1);
  __syncthreads();
  if (t_id == 0) {
    sum += 0.5 * *temp_scalar;
    //printf("E_ij = %.3e\n", 0.5 * *temp_scalar);
  }

  if (t_id == 0) {
    *out = sum;
  }
}

/**
 * Calculate the local potentials at each DB site.
 * @param n the charge configuration.
 * @param n_dbs count of DBs.
 * @param v_ij precomputed inter-DB repulsion.
 * @param v_ext external potentials at each DB site.
 * @param out output array to write to, memory must be preallocated.
 * @param temp_vec temporary vector preallocated to size n_dbs * sizeof(TFloat).
 */
template<typename TCharge, typename TFloat>
__device__ void calcLocalPotentials(TCharge *n, int n_dbs, TFloat *v_ij, TFloat *v_ext, TFloat *out, TFloat *temp_vec)
{
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  mvProd(v_ij, n, temp_vec, n_dbs, n_dbs);
  __syncthreads();

  for (int i=t_id; i<n_dbs; i+=stride) {
    out[i] = - v_ext[i] - temp_vec[i];
  }
}

/**
 * Calculate the delta update for system energy and local potentials after charge population changes.
 * @param dn the update to charge configuration.
 * @param n_dbs count of DBs.
 * @param v_ij precomputed inter-DB repulsion.
 * @param v_local pointer to the local potentials, will be updated.
 * @param E_sys pointer to the current system energy, will be updated.
 * @param temp_scalar temporary scalar preallocated to sizeof(TFloat).
 * @param temp_vec_0 temporary vector preallocated to n_dbs * sizeof(TFloat).
 * @param temp_vec_1 temporary vector preallocated to n_dbs * sizeof(TFloat).
 */
template<typename TCharge, typename TFloat>
__device__ void popChangeDeltaUpdates(TCharge *dn, int n_dbs, TFloat *v_ij, TFloat *v_local, TFloat *E_sys, TFloat *temp_scalar, TFloat *temp_vec_0, TFloat *temp_vec_1)
{
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;

  // update E_sys
  __syncthreads();
  vvInnerProd(v_local, dn, temp_scalar, n_dbs, temp_vec_0);
  __syncthreads();
  if (t_id == 0) {
    *E_sys -= *temp_scalar;
  }
  __syncthreads();
  mvProd(v_ij, dn, temp_vec_0, n_dbs, n_dbs);
  __syncthreads();
  vvInnerProd(dn, temp_vec_0, temp_scalar, n_dbs, temp_vec_1);
  __syncthreads();
  if (t_id == 0) {
    *E_sys += 0.5 * *temp_scalar;
  }

  // update v_local
  __syncthreads();
  mvProd(v_ij, dn, temp_vec_0, n_dbs, n_dbs);
  __syncthreads();
  vvScaledAdd(v_local, temp_vec_0, (TFloat) 1., (TFloat) -1., v_local, n_dbs);
}