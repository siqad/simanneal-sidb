// @file:     simanneal_alg.cu
// @author:   Samuel
// @created:  2021.07.30
// @license:  Apache License 2.0
//
// @desc:     Implementation of SimAnneal's CUDA algorithms

#include <cuda_runtime.h>
#include <cublas_v2.h>  // TODO might not use this one
#include <curand_kernel.h>
//#include <cutlass/numeric_types.h>
//#include <cutlass/gemm/kernel/gemm.h>
#include "constants.h"
#include "phys_model.cu"
//#include "gemm.cu"

// GLOBAL VARS

__device__ cublasHandle_t *cb_hdl;
__device__ int n_dbs, anneal_cycles, hop_attempt_factor;
__device__ float alpha, kT_start, kT_min, v_freeze_thresh, v_freeze_step, muzm, mupz;
__device__ float *v_ij, *v_ext;

// MACROS

/**
 * 0-based index for cuBLAS.
 */
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

/**
 * CUDA error checking.
 */
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/**
 * cuBLAS error checking.
 */
#define cublasCheckErrors(cublas_status) \
  if (cublas_status != CUBLAS_STATUS_SUCCESS) { \
    printf("Fatal cuBLAS error: %d (at %s, %d)\n", (int)(cublas_status), \
        __FILE__, __LINE__); \
  }

/**
 * Only include code if debug build.
 */
#ifndef NDEBUG
#define DEBUG_RUN(code) code
#else
#define DEBUG_RUN(code)
#endif


/**
 * Print 1D float array content in device.
 */
#define print1DArrayFloat(name, arr, size) \
  printf("%s=[", name); \
  for (int i=0; i<size; i++) { \
    printf("%f", arr[i]); \
    if (i!=size-1) \
      printf(", "); \
  } \
  printf("]\n");

/**
 * Print 1D int array content in device.
 */
#define print1DArrayInt(name, arr, size) \
  printf("%s=[", name); \
  for (int i=0; i<size; i++) { \
    printf("%i", arr[i]); \
    if (i!=size-1) \
      printf(", "); \
  } \
  printf("]\n");

/**
 * Print 2D float square array content in device in cublas indices.
 */
#define print2DArrayFloat(name, arr, size) \
  printf("%s=\n[", name); \
  for (int i=0; i<size; i++) {\
    if (i == 0) {\
      printf("[");\
    } else {\
      printf(" [");\
    }\
    for (int j=0; j<size; j++) {\
      printf("%f", arr[IDX2C(i,j,size)]);\
      if (j != size-1) \
        printf(", "); \
    }\
    if (i != size-1) {\
      printf("],\n"); \
    }\
  }\
  printf("]]\n");

// FUNCTIONS

/**
 * Minimal add function for testing. TODO remove when done.
 */
__global__ void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
    //y[i] = x[i] + y[i];
  }
}

/**
 * Generate a random integer in range [0, cap) and write to output pointer.
 * Note that cuRAND generates a float in range (0,1].
 * @param curand_state the cuRAND handle.
 * @param cap max int (exclusive).
 * @param output pointer to write generated int to.
 */
__device__ int randInt(curandState *curand_state, int cap)
{
  return static_cast<int>(ceilf(cap * curand_uniform(curand_state))) - 1;
}

/**
 * Return a probabilistic truth state.
 * Note that cuRAND generates a float in range (0,1].
 * @param curand_state the cuRAND handle.
 * @param prob the probability of returning true.
 * @param truth_state truth state to return.
 */
__device__ void evalProb(curandState *curand_state, float prob, bool *result)
{
  float rand = curand_uniform(curand_state);
  *result = rand < prob;
  //return curand_uniform(curand_state) < prob;
}

/**
 * Test math operations
 */
__global__ void testMath()
{
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;

  // test matrix multiplication
  __shared__ int r_dim_1;
  __shared__ int c_dim_1;
  __shared__ float *m1, *m2, *v1, *v2, *out, *mv_out, *vv_out;
  __shared__ float *temp_arr;
  if (t_id == 0) {
    //cudaMalloc(&m1, in_arr_size * sizeof(float));
    r_dim_1 = 3;
    c_dim_1 = 3;
    int in_arr_size = r_dim_1 * c_dim_1;
    m1 = new float[in_arr_size];
    m2 = new float[in_arr_size];
    v1 = new float[r_dim_1];
    v2 = new float[r_dim_1];
    out = new float[r_dim_1*r_dim_1];
    mv_out = new float[r_dim_1];
    vv_out = new float;
    temp_arr = new float[r_dim_1];
    m1[IDX2C(0, 0, c_dim_1)] = 1;
    m1[IDX2C(0, 1, c_dim_1)] = 2;
    m1[IDX2C(0, 2, c_dim_1)] = 3;
    m1[IDX2C(1, 0, c_dim_1)] = 4;
    m1[IDX2C(1, 1, c_dim_1)] = 5;
    m1[IDX2C(1, 2, c_dim_1)] = 6;
    m1[IDX2C(2, 0, c_dim_1)] = 7;
    m1[IDX2C(2, 1, c_dim_1)] = 8;
    m1[IDX2C(2, 2, c_dim_1)] = 9;
    m2[IDX2C(0, 0, c_dim_1)] = 1;
    m2[IDX2C(0, 1, c_dim_1)] = 2;
    m2[IDX2C(0, 2, c_dim_1)] = 3;
    m2[IDX2C(1, 0, c_dim_1)] = 4;
    m2[IDX2C(1, 1, c_dim_1)] = 5;
    m2[IDX2C(1, 2, c_dim_1)] = 6;
    m2[IDX2C(2, 0, c_dim_1)] = 7;
    m2[IDX2C(2, 1, c_dim_1)] = 8;
    m2[IDX2C(2, 2, c_dim_1)] = 9;
    print2DArrayFloat("m1", m1, r_dim_1);
    print2DArrayFloat("m2", m2, r_dim_1);
    v1[0] = 3;
    v1[1] = 6;
    v1[2] = 9;
    v2[0] = 2;
    v2[1] = 4;
    v2[2] = 6;
  }
  __syncthreads();
  mmProd2D(m1, m2, out, r_dim_1, c_dim_1);
  __syncthreads();
  if (t_id == 0) {
    print2DArrayFloat("out", out, r_dim_1);
  }
  mvProd(m1, v1, mv_out, r_dim_1, c_dim_1);
  __syncthreads();
  if (t_id == 0) {
    printf("mv_out = [%f, %f, %f]\n", mv_out[0], mv_out[1], mv_out[2]);
  }
  vvInnerProdOfMatrixExtractedArrays(m1, m2, vv_out, 1, r_dim_1, c_dim_1, temp_arr);
  __syncthreads();
  if (t_id == 0) {
    printf("vv_out = %f\n", *vv_out);
  }
  vvInnerProd(v1, v2, vv_out, r_dim_1, temp_arr);
  __syncthreads();
  if (t_id == 0) {
    printf("vv_out_2 = %f\n", *vv_out);
  }
}

/**
 * Randomize initial population
 */
template<typename TCharge>
__device__ void randomizeChargeStates(TCharge *n, int n_dbs, curandState *curand_state)
{
  int t_id = threadIdx.x + (blockIdx.x * blockDim.x);
  if (t_id >= n_dbs) {
    // skip threads that have nothing to do
    return;
  }
  int stride = blockDim.x * gridDim.x;

  for (int i=t_id; i<n_dbs; i+=stride) {
    n[i] = randInt(curand_state, 3) - 1;
  }
}

/**
 * Generate a suggested population delta for the given physical conditions.
 */
template<typename TCharge, typename TFloat>
__device__ void genPopDelta(TCharge *n, TCharge *dn, bool *changed, TFloat *v_local, TFloat *v_freeze,
                            TFloat *kT, TFloat muzm, TFloat mupz, int n_dbs,
                            curandState *curand_state)
{
  int t_id = threadIdx.x + (blockIdx.x * blockDim.x);
  int stride = blockDim.x * gridDim.x;

  if (t_id == 0) {
    *changed = false;
  }
  __syncthreads();

  float prob;
  float x;
  int change_dir;
  bool accept;

  for (int i=t_id; i<n_dbs; i+=stride) {
    if (n[i] == -1) {
      // probability from DB- to DB0
      x = - (v_local[i] + muzm) + *v_freeze;
      change_dir = 1;
    } else if (n[i] == 1) {
      // probability from DB+ to DB0
      x = v_local[i] + mupz + *v_freeze;
      change_dir = -1;
    } else {
      if (fabs(v_local[i] + muzm) < fabs(v_local[i] + mupz)) {
        // closer to DB(0/-) transition level, probability from DB0 to DB-
        x = v_local[i] + muzm + *v_freeze;
        change_dir = -1;
      } else {
        // closer to DB(+/0) transition level, probability from DB0 to DB+
        x = - (v_local[i] + mupz) + *v_freeze;
        change_dir = 1;
      }
    }
    prob = 1. / (1 + (exp(x / *kT)));

    evalProb(curand_state, prob, &accept);
    //if (evalProb(curand_state, prob)) {
    if (accept) {
      dn[i] = change_dir;
      *changed = true;
    } else {
      dn[i] = 0;
    }
  }
}

/**
 * Choose sites to hop from (charged) and to (neutral).
 * @param dbm_occ DB- occupation indices.
 * @param db0_occ DB0 occupation indices.
 * @param dbp_occ DB+ occupation indices.
 * @param dbm_count count of DBs in -ve charge state.
 * @param db0_count count of DBs in neutral charge state.
 * @param dbp_count count of DBs in +ve charge state.
 * @param from_state charge state of the "from" DB.
 * @param from_occ_ind index of the "from" DB on the corresponding occ array.
 * @param to_occ_ind index of the "to" DB on the neutral occ array.
 * @param from_db_ind index of the DB site that the charge is hopping from.
 * @param to_db_ind index of the DB site that the charge is hopping to.
 * @param curand_state cuRAND state pointer for RNG.
 */
__device__ void chooseHopIndices(int *dbm_occ, int *db0_occ, int *dbp_occ,
                                 int dbm_count, int db0_count, int dbp_count,
                                 int *from_state, int *from_occ_ind,
                                 int *to_occ_ind, int *from_db_ind,
                                 int *to_db_ind, curandState *curand_state)
{
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // only need threads 0 and 1 to generate from and to indices
  if (t_id >= 2) {
    return;
  }

  int cap;
  for (int i=t_id; i<2; i+=stride) {
    if (i == 0) {
      // from index
      cap = dbm_count + dbp_count;
    } else {
      // to index
      cap = db0_count;
    }
    int hop_ind = randInt(curand_state, cap);

    if (i == 0) {
      // set from index
      if (hop_ind < dbm_count) {
        *from_state = -1;
        *from_occ_ind = hop_ind;
        *from_db_ind = dbm_occ[*from_occ_ind];
      } else {
        *from_state = 1;
        *from_occ_ind = hop_ind - dbm_count;
        *from_db_ind = dbp_occ[*from_occ_ind];
      }
    } else {
      // set to index
      *to_occ_ind = hop_ind;
      *to_db_ind = db0_occ[hop_ind];
    }
  }
}

/**
 * Calculate the energy delta for a hop operation.
 * @param n the charge states.
 * @param n_dbs count of DBs.
 * @param from_db_ind hopping from DB at this index.
 * @param to_db_ind hopping to DB at this index.
 * @param v_local local potential at DB sites.
 * @param v_ij pre-computed coulombic repulsion.
 * @param E_delta return change in energy resulting from this hop.
 */
template <typename TCharge, typename TFloat>
__device__ void calcHopEnergyDelta(TCharge *n, int n_dbs, int *from_state,
                                   int *from_db_ind, int *to_db_ind,
                                   TFloat *v_local, TFloat *v_ij, TFloat *E_delta)
{
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_id != 0) {
    return;
  }

  int dn_i = - *from_state;
  int dn_j = *from_state;
  *E_delta = -v_local[*from_db_ind] * dn_i - v_local[*to_db_ind] * dn_j - v_ij[IDX2C(*from_db_ind, *to_db_ind, n_dbs)];
}

/**
 * Return whether a hop should be accepted at this stage of annealing.
 * @param E_delta change in energy resulting from this hop.
 * @param kT current annealing temperature.
 * @param accept return boolean indicating whether this is accepted.
 * @param curand_state cuRAND handle.
 */
template <typename TFloat>
__device__ void acceptHop(TFloat *E_delta, TFloat *kT, bool *accept, curandState *curand_state)
{
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_id != 0) {
    return;
  }

  if (*E_delta < 0) {
    *accept = true;
  } else {
    TFloat prob = exp(-(*E_delta) / (*kT));
    //*accept = evalProb(curand_state, prob);
    evalProb(curand_state, prob, accept);
  }
}

/**
 * Perform the specified hop and update relevant parameters.
 * @param n charge configuration.
 * @param n_dbs DB count.
 * @param from_db_ind originate DB site.
 * @param to_db_ind destination DB site.
 * @param E_sys system energy to be updated.
 * @param E_delta previously calculated change in system energy from the hop.
 * @param v_local local potentials to be updated.
 * @param v_ij precomputed coulombic repulsion.
 */
template <typename TCharge, typename TFloat>
__device__ void performHopUpdates(TCharge *n, int n_dbs, int *from_state,
                                  int *from_db_ind, int *to_db_ind,
                                  TFloat *E_sys, TFloat *E_delta,
                                  TFloat *v_local, TFloat *v_ij)
{
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  if (t_id == 0) {
    n[*from_db_ind] += - *from_state;
    n[*to_db_ind] += *from_state;
    *E_sys += *E_delta;
  }

  for (int i=t_id; i<n_dbs; i+=stride) {
    v_local[i] = v_local[i] - ((- *from_state) * v_ij[IDX2C(i, *from_db_ind, n_dbs)] + (*from_state) * v_ij[IDX2C(i, *to_db_ind, n_dbs)]);
  }
}

/**
 * Run the main SimAnneal algorithm on the GPU.
 * 
 * @param stream_id CUDA stream ID.
 * @param n_out charge states output.
 */
__global__ void runAnneal(int stream_id, int *n_out)
{
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;

  // RNG
  curandState curand_state;
  curand_init((unsigned long long)clock() + t_id, 0, 0, &curand_state);

  // best charge state tracking
  float *best_n;      // current best charge configuration
  float best_E;       // current best system energy

  // copy arrays over from global memory
  __shared__ float *v_ij_l, *v_ext_l;
  if (t_id == 0) {
    v_ij_l = new float[n_dbs*n_dbs];
    v_ext_l = new float[n_dbs];
    for (int i=0; i<n_dbs*n_dbs; i++) {
      v_ij_l[i] = v_ij[i];
    }
    for (int i=0; i<n_dbs; i++) {
      v_ext_l[i] = v_ext_l[i];
    }
  }

  // charge state, population, and env tracking
  __shared__ float *n, *dn;             // current/delta DB site charge state (float due to linalg)
  __shared__ int dbm_occ_count, db0_occ_count, dbp_occ_count;
  __shared__ int *dbm_occ, *db0_occ, *dbp_occ;  // DB indices currently at -ve/neutral/+ve charge state
  __shared__ bool *pop_changed;         // whether population is updated
  __shared__ float *E_sys, *E_delta;
  __shared__ float *v_local;            // local potential at each site
  if (t_id == 0) {
    n = new float[n_dbs];
    dn = new float[n_dbs];
    dbm_occ = new int[n_dbs];
    db0_occ = new int[n_dbs];
    dbp_occ = new int[n_dbs];
    pop_changed = new bool;
    E_sys = new float;
    E_delta = new float;
    v_local = new float[n_dbs];
    for (int i=0; i<n_dbs; i++) {
      n[i] = 0.;
      dn[i] = 0.;
      v_local[i] = 0.;
    }
  }

  // hop tracking
  __shared__ int hop_attempts, max_hop_attempts;
  __shared__ int *from_state, *from_occ_ind, *to_occ_ind, *from_db_ind, *to_db_ind;
  __shared__ bool *accept_hop;
  if (t_id == 0) {
    from_state = new int;
    from_occ_ind = new int;
    to_occ_ind = new int;
    from_db_ind = new int;
    to_db_ind = new int;
    accept_hop = new bool;
  }
  // TODO: initialize above

  // annealing cycle tracking
  __shared__ int cycle; // current anneal cycle
  __shared__ float *v_freeze, *kT;
  if (t_id == 0) {
    cycle=0;  // current anneal cycle
    v_freeze = new float;
    kT = new float;
    *v_freeze = 0.;
    *kT = kT_start; // TODO reenable
  }

  // temp vars
  __shared__ float *temp_scalar;
  __shared__ float *temp_scalar_1;
  __shared__ float *temp_vec_ndbs_0;
  __shared__ float *temp_vec_ndbs_1;
  if (t_id == 0) {
    temp_scalar = new float;
    temp_scalar_1 = new float;
    temp_vec_ndbs_0 = new float[n_dbs];
    temp_vec_ndbs_1 = new float[n_dbs];
  }

  // randomize initial charge state and find energy
  __syncthreads();
  randomizeChargeStates(n, n_dbs, &curand_state);
  __syncthreads();
  calcSystemEnergy(n, n_dbs, v_ij_l, v_ext_l, E_sys, temp_scalar, temp_vec_ndbs_0, temp_vec_ndbs_1);
  calcLocalPotentials(n, n_dbs, v_ij_l, v_ext_l, v_local, temp_vec_ndbs_0);
  __syncthreads();
  if (t_id == 0) {
    print1DArrayFloat("randomized n", n, n_dbs);
    printf("E_sys=%.3e\n", *E_sys);
    print1DArrayFloat("v_local", v_local, n_dbs);
  }

  // test system energy calc
  /*
  n[0] = -1;
  n[1] = 0;
  n[2] = 0;
  n[3] = -1;
  n[4] = -1;
  n[5] = 0;
  n[6] = -1;
  __shared__ float *energy;
  if (t_id == 0) {
    energy = new float;
  }
  __syncthreads();
  calcSystemEnergy(n, n_dbs, v_ij, v_ext, energy);
  printf("energy=%.3e\n", *energy);
  */

  // run algorithm
  if (t_id == 0) {
    printf("***** Begin SimAnneal algorithm *****\n\n");
  }
  //anneal_cycles = 10; // TODO: remove
  while (cycle < anneal_cycles) {
    // Population update
    genPopDelta(n, dn, pop_changed, v_local, v_freeze, kT, muzm, mupz, n_dbs, &curand_state);
    __syncthreads();
    DEBUG_RUN(
      if (t_id == 0) {
        print1DArrayFloat("dn", dn, n_dbs); // TODO: remove
      }
    )
    if (pop_changed) {
      vvAdd(n, dn, n, n_dbs); // update the charge list
      popChangeDeltaUpdates(dn, n_dbs, v_ij_l, v_local, E_sys, temp_scalar, temp_vec_ndbs_0, temp_vec_ndbs_1);
      __syncthreads();
      DEBUG_RUN(
        if (t_id == 0) {
          print1DArrayFloat("new n", n, n_dbs); // TODO: remove
          printf("E_sys=%.3e\n", *E_sys); // TODO: remove
        }
      )
      DEBUG_RUN(
        calcSystemEnergy(n, n_dbs, v_ij_l, v_ext_l, temp_scalar, temp_scalar_1, temp_vec_ndbs_0, temp_vec_ndbs_1);
        if (t_id == 0) {
          printf("E_sys_calc=%.3e\n", *temp_scalar);
          print1DArrayFloat("v_local", v_local, n_dbs); // TODO: remove
        }
      )

      // update occupation lists
      if (t_id == 0) {
        int dbm_ind=0, db0_ind=0, dbp_ind=0;
        for (int db_ind=0; db_ind<n_dbs; db_ind++) {
          if (n[db_ind] == -1) {
            dbm_occ[dbm_ind++] = db_ind;
          } else if (n[db_ind] == 0) {
            db0_occ[db0_ind++] = db_ind;
          } else {
            dbp_occ[dbp_ind++] = db_ind;
          }
        }
        dbm_occ_count = dbm_ind;
        db0_occ_count = db0_ind;
        dbp_occ_count = dbp_ind;
        DEBUG_RUN(
          if (t_id == 0) {
            printf("DB-: %d, DB0: %d, DB+: %d\n", dbm_occ_count, db0_occ_count, dbp_occ_count);
            print1DArrayInt("dbm_occ", dbm_occ, n_dbs);
            print1DArrayInt("db0_occ", db0_occ, n_dbs);
            print1DArrayInt("dbp_occ", dbp_occ, n_dbs);
          }
        )
      }
    }

    __syncthreads();
    if (t_id == 0) {
      hop_attempts = 0;
      max_hop_attempts = 0;
      if (dbm_occ_count + dbp_occ_count < n_dbs && db0_occ_count < n_dbs) {
        max_hop_attempts = max(dbm_occ_count+dbp_occ_count, db0_occ_count);
        max_hop_attempts *= hop_attempt_factor;
      }
    }

    // Hopping
    __syncthreads();
    while (hop_attempts < max_hop_attempts) {
      chooseHopIndices(dbm_occ, db0_occ, dbp_occ, dbm_occ_count, db0_occ_count,
                       dbp_occ_count, from_state, from_occ_ind, to_occ_ind,
                       from_db_ind, to_db_ind, &curand_state);
      __syncthreads();
      DEBUG_RUN(
        if (t_id == 0) {
          printf("Chose to hop from %d to %d (from state is %d)\n", *from_db_ind, *to_db_ind, *from_state);
        }
      )
      calcHopEnergyDelta(n, n_dbs, from_state, from_db_ind, to_db_ind, v_local, v_ij_l, E_delta);
      __syncthreads();
      acceptHop(E_delta, kT, accept_hop, &curand_state);
      __syncthreads();
      if (*accept_hop) {
        DEBUG_RUN(
          if (t_id == 0) {
            printf("ACCEPTED\n");
          }
        )
        // hop accepted, update telemetry
        performHopUpdates(n, n_dbs, from_state, from_db_ind, to_db_ind, E_sys, E_delta, v_local, v_ij_l);
        __syncthreads();
        DEBUG_RUN(
          if (t_id == 0) {
            print1DArrayFloat("n after hop", n, n_dbs);
          }
        )
        if (t_id == 0) {
          if (*from_state == -1) {
            dbm_occ[*from_occ_ind] = *to_db_ind;
          } else {
            dbp_occ[*from_occ_ind] = *to_db_ind;
          }
          db0_occ[*to_occ_ind] = *from_db_ind;
        }
      }

      if (t_id == 0) {
        hop_attempts++;
      }
      __syncthreads();
    }


    // Annealing schedule update
    if (t_id == 0)
    {
      // TODO: annealing schedule parameter update
      cycle++;
      *kT = kT_min + (*kT - kT_min) * alpha;
      if (*v_freeze < v_freeze_thresh) {
        *v_freeze += v_freeze_step;
      }
      DEBUG_RUN(printf("\n\n");)
    }
    __syncthreads();
  }

  if (t_id == 0) {
    printf("Final configuration:\n");
    print1DArrayFloat("n", n, n_dbs);
    printf("Final energy: %.3e\n", *E_sys);
  }

  // test RNG
  /*
  for (int i=0; i<20; i++) {
    int num = randInt(&curand_state, 20);
    printf("Rand int: %d\n", num);
  }
  */

  // free memory
  if (t_id == 0) {
    printf("Free pointer memories...\n");
    free(v_ij_l);
    free(v_ext_l);
    free(n);
    free(dn);
    free(dbm_occ);
    free(db0_occ);
    free(dbp_occ);
    free(pop_changed);
    free(E_sys);
    free(E_delta);
    free(v_local);
    free(from_state);
    free(from_occ_ind);
    free(to_occ_ind);
    free(from_db_ind);
    free(to_db_ind);
    free(accept_hop);
    free(v_freeze);
    free(kT);
    free(temp_scalar);
    free(temp_scalar_1);
    free(temp_vec_ndbs_0);
    free(temp_vec_ndbs_1);
    // TODO: some new variables aren't freed here, check back
    printf("Finished destroying pointers.\n\n");
  }
}

__global__ void initVij(int n_dbs, float eps_r, float debye_length, float *db_locs)
{
  int t_id = threadIdx.x + (blockIdx.x * blockDim.x);
  int stride = blockDim.x * gridDim.x;
  
  float Kc = 1./(4. * constants::PI * eps_r * constants::EPS0);
  printf("Initializing v_ij\n");
  printf("t_id=%d, stride=%d, n_dbs=%d\n", t_id, stride, n_dbs);
  for (int i=t_id; i<n_dbs; i+=stride) {
    for (int j=i; j<n_dbs; j++) {
      if (i==j) {
        v_ij[IDX2C(i,j,n_dbs)] = 0;
        continue;
      }
      float r = sqrtf( powf(fabsf(db_locs[IDX2C(i,0,n_dbs)] - db_locs[IDX2C(j,0,n_dbs)]),2) 
                      +powf(fabsf(db_locs[IDX2C(i,1,n_dbs)] - db_locs[IDX2C(j,1,n_dbs)]),2) );
      r *= powf(10, -10); // convert angstrom to m
      v_ij[IDX2C(i,j,n_dbs)] = constants::Q0 * Kc * expf(-r/(debye_length*1e-9)) / r;
      v_ij[IDX2C(j,i,n_dbs)] = v_ij[IDX2C(i,j,n_dbs)];
      printf("r(%d,%d)=%.3e, v_ij[%d,%d]=%.3e\n", i, j, r, i, j, v_ij[IDX2C(i,j,n_dbs)]);
    }
  }
}

/**
 * Initialize device variables
 */
__global__ void initDeviceVars(int t_n_dbs, float t_muzm, float t_mupz, float t_alpha, float kT_start_in,
                               float kT_min_in, float t_v_freeze_thresh, float v_freeze_step_in, 
                               int t_anneal_cycles, int t_hop_attempt_factor,
                               float *t_v_ext)
{
  // assign simple variables
  printf("Assigning variables\n");
  n_dbs = t_n_dbs;
  muzm = t_muzm;
  mupz = t_mupz;
  alpha = t_alpha;
  kT_start = kT_start_in;
  kT_min = kT_min_in;
  v_freeze_thresh = t_v_freeze_thresh;
  v_freeze_step = v_freeze_step_in;
  anneal_cycles = t_anneal_cycles;
  hop_attempt_factor = t_hop_attempt_factor;

  //cudaMalloc(&v_ext, n_dbs*sizeof(float));
  v_ext = new float[n_dbs];
  for (int i=0; i<n_dbs; i++) {
    v_ext[i] = t_v_ext[i];
  }

  //// calculate v_ij using db_locs
  //printf("Initializing v_ij\n");
  //cudaMalloc(&v_ij, n_dbs*n_dbs*sizeof(float));
  v_ij = new float[n_dbs*n_dbs];

  //__syncthreads();

  //printf("Device vars initialized\n");
}

///*
__global__ void cleanUpDeviceVars(int num_streams)
{
  //for (int i=0; i<num_streams; i++) {
  //  cublasCheckErrors(cublasDestroy_v2(cb_hdl[i]));
  //}
  free(v_ij);
  free(v_ext);
}

__device__ void initVLocal(cublasHandle_t cublas_handle, int n_dbs, float *n, float *v_ext, float *v_local)
{
  // v_local = v_ext - dot(v_ij, n)
  for (int i=0; i<n_dbs; i++) {
    v_local[i] = 0;
  }

  // dot(v_ij, n)
  float alpha=1;
  float beta=0;
  cublasCheckErrors(cublasSgemv(cublas_handle, CUBLAS_OP_N, n_dbs, n_dbs, &alpha, v_ij, n_dbs, n, 1, 
      &beta, v_local, 1));
  __syncthreads();

  // v_ext - above
  alpha=-1;
  cublasCheckErrors(cublasSaxpy(cublas_handle, n_dbs, &alpha, v_local, 1, v_ext, 1));
  __syncthreads();
}
//*/