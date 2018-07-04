// @file:     sim_anneal.cu
// @author:   Samuel
// @created:  2017.08.23
// @editted:  2018.06.13 - Robert
// @license:  GNU LGPL v3
//
// @desc:     Simulated annealing physics engine

#include "sim_anneal.h"
#include <ctime>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>


//#define STEADY_THREASHOLD 700       //arbitrary value used in restarting

using namespace phys;

std::mutex siqadMutex;

__device__ cublasHandle_t *cb_hdl;
__device__ int n_dbs;
__device__ float mu, kT_start, kT0, kT_step, v_freeze_step, t_max;
__device__ float *v_ij, *v_ext;

// CUDA error checking
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// cuBLAS error checking
#define cublasCheckErrors(cublas_status) \
  if (cublas_status != CUBLAS_STATUS_SUCCESS) { \
    printf("Fatal cuBLAS error: %d (at %s, %d)\n", (int)(cublas_status), \
        __FILE__, __LINE__); \
  }

// print 1d float array content in device
#define print1DArrayFloat(name, arr, size) \
  printf("%s=[", name); \
  for (int i=0; i<size; i++) { \
    printf("%f", arr[i]); \
    if (i!=size-1) \
      printf(", "); \
  } \
  printf("]\n");

// print 1d int array content in device
#define print1DArrayInt(name, arr, size) \
  printf("%s=[", name); \
  for (int i=0; i<size; i++) { \
    printf("%i", arr[i]); \
    if (i!=size-1) \
      printf(", "); \
  } \
  printf("]\n");

// print 2d float square array content in device in cublas indices
#define print2DArrayFloat(name, arr, size) \
  printf("%s=[", name); \
  for (int i=0; i<size; i++) {\
    for (int j=0; j<size; j++) {\
      printf("%f", arr[IDX2C(i,j,size)]);\
      if (j != size-1) \
        printf(", "); \
    }\
    if (i != size-1) \
      printf(",\n"); \
  }\
  printf("]\n");

// *v_freeze and *kT are shared within the stream, pre-initialize them.
// *pop_changed should be an array of bools with num_threads elements.
// *E_sys should be an array of floats with num_threads elements.
// *n, *dn and *v_local should be arrays of floats with num_thread*n_dbs elements.
// TODO add float *n_g back in for final passing back
__global__ void simAnnealParallel(int stream_id, int results_to_return, float *n_out)
{
  // TODO add n_dbs, *v_ext, t_max and kT_init to device memory. Maybe even add n_dbs to shared memory for testing
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);

  // RNG
  curandState curand_state;
  curand_init((unsigned long long)clock() + tId, 0, 0, &curand_state);

  /*cublasHandle_t cublas_handle;
  cublasCheckErrors(cublasCreate_v2(&cublas_handle));*/
  cublasHandle_t cublas_handle = cb_hdl[stream_id];

  // cycle tracking
  int t=0;                // current anneal cycle
  float *v_freeze, *kT;
  cudaMalloc(&v_freeze, sizeof(float));
  cudaMalloc(&kT, sizeof(float));
  *v_freeze=0;
  *kT=kT_start;

  // result return tracking
  int t_return_results = t_max - results_to_return;
  int return_i = tId * n_dbs;   // start returning at this index
  
  // population related vars
  int n_elec, n_empty;    // number of electrons, number of vacant DBs
  float *n;               // current occupation of DB sites
  float *dn;              // change of occupation for population update
  float *v_local;         // local potential at each site
  bool *pop_changed;      // indicate whether population has changed during this cycle
  cudaMalloc(&n, n_dbs*sizeof(float));
  cudaMalloc(&dn, n_dbs*sizeof(float));
  cudaMalloc(&v_local, n_dbs*sizeof(float));
  cudaMalloc(&pop_changed, sizeof(bool));
  for (int i=0; i<n_dbs; i++) {
    n[i] = 0.;
    dn[i] = 0.;
    v_local[i] = 0.;
  }

  // hop related vars
  int hop_attempts;
  int *occ;               // first n_elec elements are indices of occupied sites; the rest are unoccupied indices
  int from_ind, to_ind;
  int from_occ_ind, to_occ_ind;
  bool accept_hop;
  cudaMalloc(&occ, n_dbs*sizeof(int));

  // initialize system and local energies
  //printf("Initializing system energy and local energy\n");
  float E_sys, E_del;
  systemEnergy(cublas_handle, n_dbs, n, v_ext, &E_sys);
  initVLocal(cublas_handle, n_dbs, n, v_ext, v_local);

  while (t < t_max) {
    // Population
    *pop_changed = false;
    //genPopulationDelta(&curand_state, n_dbs, n, v_local, v_freeze, kT, dn, pop_changed);
    // TODO change fixed 64 threads to dynamic number dependent on n_dbs
    genPopulationDelta<<<1,64>>>(n_dbs, n, v_local, v_freeze, kT, dn, pop_changed);
    cudaDeviceSynchronize();
    if (*pop_changed) {
      // n + dn
      float alpha=1;
      cublasCheckErrors(cublasSaxpy(cublas_handle, n_dbs, &alpha, dn, 1, n, 1));

      // E_sys += energy delta from population change
      populationChangeEnergyDelta(cublas_handle, n_dbs, dn, v_local, &E_del);
      E_sys += E_del;

      // v_local = - prod(v_ij, dn) + v_local 
      alpha=-1;
      float beta=1;
      cublasCheckErrors(cublasSgemv(cublas_handle, CUBLAS_OP_N, n_dbs, n_dbs, &alpha, 
            v_ij, n_dbs, dn, 1, &beta, v_local, 1));

      //print1DArrayFloat("Population changed, dn", dn, n_dbs);
      //print1DArrayFloat("v_local", v_local, n_dbs);
      //print1DArrayFloat("n", n, n_dbs);

      // occupation list update
      int occ_ind=0, unocc_ind=n_dbs-1;
      for (int db_ind=0; db_ind<n_dbs; db_ind++)
        n[db_ind] == 1.0 ? occ[occ_ind++] = db_ind : occ[unocc_ind--] = db_ind;
      n_elec = occ_ind;
      //print1DArrayInt("occ", occ, n_dbs);
    }
    cudaDeviceSynchronize();

    // Hopping
    if (n_elec != 0 && n_elec != n_dbs) {
      //hopped=false;
      hop_attempts=0;
      int max_hops = (n_dbs-n_elec)*3;  
      n_empty = n_dbs-n_elec;
      while (hop_attempts < max_hops) {
        randInt(&curand_state, n_elec, &from_occ_ind);
        randInt(&curand_state, n_empty, &to_occ_ind);
        to_occ_ind += n_elec;
        from_ind = occ[from_occ_ind];
        to_ind = occ[to_occ_ind];

        //printf("from_occ_ind=%d, to_occ_ind=%d\n", from_occ_ind, to_occ_ind);
        hopEnergyDelta(from_ind, to_ind, n_dbs, v_local, &E_del);
        acceptHop(&curand_state, &E_del, kT, &accept_hop);

        //printf("Attempting hop from sites %d to %d with E_del=%f...\n", from_ind, to_ind, E_del);
        if (accept_hop) {
          //hopped = E_del != 0.; // treat degenerate state change as steady
          // update relevant arrays to keep track of hops
          n[from_ind] = 0.;
          n[to_ind] = 1.;
          occ[from_occ_ind] = to_ind;
          occ[to_occ_ind] = from_ind;
          //printf("Post hop:\nn[%d]=%f, n[%d]=%f\nocc[%d]=%d,occ[%d]=%d\n", from_ind, n[from_ind], to_ind, n[to_ind], from_occ_ind, occ[from_occ_ind], to_occ_ind, occ[to_occ_ind]);

          // update energy
          E_sys += E_del;
          updateVLocal<<<1,64>>>(from_ind, to_ind, n_dbs, v_local);
          //print1DArrayFloat("Accepted. New v_local=", v_local, n_dbs);
          //print1DArrayFloat("New n=", n, n_dbs);
        }
        cudaDeviceSynchronize();
        hop_attempts++;
      }
    }
    cudaDeviceSynchronize();

    // write result to return array if the schedule is near the end
    if (t >= t_return_results) {
      for (int i=0; i<n_dbs; i++) {
        n_out[i+return_i] = n[i];
      }
      return_i += n_dbs;
    }

    // time step
    timeStep(&t, kT, v_freeze);
    //printf("Cycle: %d, ending energy: %f\n\n", t, E_sys);
  }
  // wait for other threads to finish
  //cudaDeviceSynchronize();

  /*print1DArrayFloat("Final n\n", n, n_dbs);
  printf("Ending energy (delta): %f\n", E_sys);

  systemEnergy(n_dbs, n, v_ext, &E_sys);
  printf("Ending energy (actua): %f\n", E_sys);*/

  float E_sys_actual;
  systemEnergy(cublas_handle, n_dbs, n, v_ext, &E_sys_actual);
  printf("tId=%d\t\tn_elec=%d\tDelta energy=%f\tActual energy: %f\n", tId, n_elec, E_sys, E_sys_actual);

  returnN(n, n_out);

  //cublasCheckErrors(cublasDestroy_v2(cublas_handle));
  free(v_freeze);
  free(kT);
  free(n);
  free(dn);
  free(v_local);
  free(pop_changed);
  free(occ);
}

// one handle per stream
__global__ void initCublasHandles(int num_streams)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  int stride = blockDim.x * gridDim.x;
  if (tId == 0)
    cudaMalloc(&cb_hdl, num_streams*sizeof(cublasHandle_t));
  cudaDeviceSynchronize();

  for (int i=tId; i<num_streams; i+=stride)
    cublasCheckErrors(cublasCreate_v2(&cb_hdl[i]));
  cudaDeviceSynchronize();
}

__global__ void destroyCublasHandles(int num_streams) {
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  int stride = blockDim.x * gridDim.x;
  for (int i=tId; i<num_streams; i+=stride) {
    cublasCheckErrors(cublasDestroy_v2(cb_hdl[i]));
  }
}

__global__ void initDeviceVars(int num_streams, float n_dbs_in, float debye_length, float mu_in, 
    float kT_start_in, float kT0_in, float kT_step_in, float v_freeze_step_in, int t_max_in, 
    float *v_ext_in, float *db_locs)
{
  // create cublas handle
  /*printf("Initializing cublas handle\n");
  cudaMalloc(&cb_hdl, num_streams*sizeof(cublasHandle_t));
  for (int i=0; i<num_streams; i++) {
    cublasCreate_v2(&cb_hdl[i]);
  }*/

  // assign simple variables
  printf("Assigning variables\n");
  n_dbs = n_dbs_in;
  mu = mu_in;
  kT_start = kT_start_in;
  kT0 = kT0_in;
  kT_step = kT_step_in;
  v_freeze_step = v_freeze_step_in;
  t_max = t_max_in;

  cudaMalloc(&v_ext, n_dbs*sizeof(float));
  for (int i=0; i<n_dbs; i++) {
    v_ext[i] = v_ext_in[i];
  }

  // calculate v_ij using db_locs
  printf("Initializing v_ij\n");
  cudaMalloc(&v_ij, n_dbs*n_dbs*sizeof(float));
  initVij<<<1,n_dbs>>>(n_dbs, debye_length, db_locs, v_ij);
  __syncthreads();

  printf("Device vars initialized\n");
}

__global__ void initVij(int n_dbs, float debye_length, float *db_locs, float *v_ij)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  int stride = blockDim.x * gridDim.x;
  
  float Kc = 1./(4. * constants::PI * constants::EPS_SURFACE * constants::EPS0);
  //for (int i=tId; i<n_dbs; i++) {
  for (int i=tId; i<n_dbs; i+=stride) {
    for (int j=i; j<n_dbs; j++) {
      if (i==j) {
        v_ij[IDX2C(i,j,n_dbs)] = 0;
        continue;
      }
      float r = sqrtf( powf(fabsf(db_locs[IDX2C(i,0,n_dbs)] - db_locs[IDX2C(j,0,n_dbs)]),2) 
                      +powf(fabsf(db_locs[IDX2C(i,1,n_dbs)] - db_locs[IDX2C(j,1,n_dbs)]),2) );
      v_ij[IDX2C(i,j,n_dbs)] = constants::Q0 * Kc * erff(r/constants::ERFDB) * expf(-r/debye_length) / r;
      v_ij[IDX2C(j,i,n_dbs)] = v_ij[IDX2C(i,j,n_dbs)];
      //printf("r[%d,%d]=%.10e, v_ij[%d,%d]=%f\n", i, j, r, i, j, v_ij[IDX2C(i,j,n_dbs)]);
    }
  }
}

__global__ void cleanUpDeviceVars(int num_streams)
{
  /*for (int i=0; i<num_streams; i++) {
    cublasCheckErrors(cublasDestroy_v2(cb_hdl[i]));
  }*/
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

__global__ void updateVLocal(int from_ind, int to_ind, int n_dbs, float *v_local)
{
  //v_local += v_ij(from-th col) - v_ij(to-th col);

  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  int stride = blockDim.x * gridDim.x;

  for (int i=tId; i<n_dbs; i+=stride) {
  //for (int i=0; i<n_dbs; i++) {
    v_local[i] += v_ij[IDX2C(i, from_ind, n_dbs)] - v_ij[IDX2C(i, to_ind, n_dbs)];
  }
}

__global__ void genPopulationDelta(int n_dbs, float *n, float *v_local, 
    float *v_freeze, float *kT, float *dn, bool *pop_changed)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  int stride = blockDim.x * gridDim.x;

  curandState curand_state;
  curand_init((unsigned long long)clock() + tId, 0, 0, &curand_state);

  //printf("Generating population delta. v_freeze=%f, kT=%f, mu=%f\n", *v_freeze, *kT, mu);
  //printf("rand_num=[");
  for (int i=tId; i<n_dbs; i+=stride) {
    // TODO consider replacing expf with __expf for faster perf
    float prob = 1. / ( 1. + __expf( ((2.*n[i]-1.)*(v_local[i]+mu) + *v_freeze ) / *kT ));
    float rand_num = curand_uniform(&curand_state);
    if (rand_num < prob) {
      dn[i] = 1. - 2.*n[i];
      *pop_changed = true;
    } else {
      dn[i] = 0.;
    }
    /*printf("%f", rand_num);
    if (i != n_dbs-1)
      printf(", ");*/
  }
  //printf("]\n");
}

__device__ void genPopulationDelta(curandState *curand_state, int n_dbs, float *n, float *v_local, 
    float *v_freeze, float *kT, float *dn, bool *pop_changed)
{
  //printf("tId=%d\t\tGenerating population delta. v_freeze=%f, kT=%f, mu=%f\n", tId, *v_freeze, *kT, mu);
  //printf("rand_num=[");
  for (int i=0; i<n_dbs; i++) {
    // TODO consider replacing expf with __expf for faster perf
    float prob = 1. / ( 1. + __expf( ((2.*n[i]-1.)*(v_local[i]+mu) + *v_freeze ) / *kT ));
    float rand_num = curand_uniform(curand_state);
    if (rand_num < prob) {
      dn[i] = 1. - 2.*n[i];
      *pop_changed = true;
    } else {
      dn[i] = 0.;
    }
    /*printf("%f", rand_num);
    if (i != n_dbs-1)
      printf(", ");*/
  }
  //printf("]\n");
}

// Total system energy including Coulombic repulsion and external voltage.
// NOTE keep this version around for benchmarking, this seems to be faster than cublas for some reason...
/*__global__ void systemEnergy(float *v, int n_dbs, float *n, float *v_ext, float *v_ij)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i=index; i<n_dbs; i+=stride) {
    *v -= v_ext[i] * n[i];
    for (int j=i+1; j<n_dbs; j++)
      *v += v_ij[i*n_dbs + j] * n[i] * n[j];
  }
}*/

__device__ void systemEnergy(cublasHandle_t cublas_handle, int n_dbs, float *n, float *v_ext, float *output)
{
  // TODO might be able to merge this function with population change energy delta with the similarity
  float *coulomb_v;
  cudaMalloc(&coulomb_v, sizeof(float));
  totalCoulombPotential(cublas_handle, n_dbs, n, coulomb_v);
  __syncthreads();
  cublasCheckErrors(cublasSdot(cublas_handle, n_dbs, v_ext, 1, n, 1, output));
  __syncthreads();
  *output *= -1;
  *output += *coulomb_v;
  free(coulomb_v);
}

// Energy change from population change.
__device__ void populationChangeEnergyDelta(cublasHandle_t cublas_handle, int n_dbs, float *dn, float *v_local, float *output)
{
  // delta E = -1 * dot(v_local, dn) + dn^T * V_ij * dn
  float *coulomb_v = (float*)malloc(sizeof(float));
  totalCoulombPotential(cublas_handle, n_dbs, dn, coulomb_v);
  cublasCheckErrors(cublasSdot(cublas_handle, n_dbs, v_local, 1, dn, 1, output));
  *output *= -1;
  *output += *coulomb_v;
  free(coulomb_v);
}

// Total potential from Coulombic repulsion in the system.
__device__ void totalCoulombPotential(cublasHandle_t cublas_handle, int n_dbs, float *n, float *v)
{
  float alpha=0.5;
  float beta=0;
  float *v_temp;
  cudaMalloc(&v_temp, n_dbs*sizeof(float));
  cublasCheckErrors(cublasSgemv(cublas_handle, CUBLAS_OP_N, n_dbs, n_dbs, &alpha, v_ij, n_dbs, n, 1, &beta, v_temp, 1));
  __syncthreads();
  cublasCheckErrors(cublasSdot(cublas_handle, n_dbs, n, 1, v_temp, 1, v));
  __syncthreads();
  free(v_temp);
}

__device__ void acceptHop(curandState *curand_state, float *v_diff, float *kT, bool *accept)
{
  if (*v_diff <= 0.) {
    *accept = true;
  } else {
    float prob = __expf( -(*v_diff) / (*kT) );
    *accept = curand_uniform(curand_state) < prob;
  }
}

__device__ void hopEnergyDelta(int i, int j, int n_dbs, float *v_local, float *v_del)
{
  *v_del = v_local[i] - v_local[j] - v_ij[IDX2C(i,j,n_dbs)];
}

__device__ void randInt(curandState *curand_state, int cap, int *output)
{
  *output = cap;
  while (*output == cap)
    *output = static_cast<int>(cap * curand_uniform(curand_state));  // floor by cast
}

__device__ void timeStep(int *t, float *kT, float *v_freeze)
{
  *t += 1;
  *kT = kT0 + (*kT - kT0) * kT_step;
  *v_freeze = (float)(*t) * v_freeze_step;
}

__device__ void returnN(float *n_local, float *n_out)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  int offset = tId*n_dbs;
  for (int i=0; i<n_dbs; i++) {
    n_out[i+offset] = n_local[i];
  }
}



//Global method for writing to vectors (global in order to avoid thread clashing).
void writeStore(SimAnneal *object, int threadId){
  siqadMutex.lock();

  object->chargeStore[threadId] = object->db_charges;
  object->energyStore[threadId] = object->config_energies; object->numElecStore[threadId] = object->n_elec;

  siqadMutex.unlock();
}

SimAnneal::SimAnneal(const int thread_id)
{
  rng.seed(std::time(NULL)*thread_id+4065);
  dis01 = boost::random::uniform_real_distribution<float>(0,1);
  threadId = thread_id;
}

void SimAnneal::runSim()
{
  // initialize variables & perform pre-calculation
  kT = 300*constants::Kb;    // kT = Boltzmann constant (eV/K) * 298 K
  v_freeze = 0;

  // resize vectors
  v_local.resize(n_dbs);

  //db_charges.resize(result_queue_size);
  n.resize(n_dbs);
  occ.resize(n_dbs);

  //config_energies.resize(result_queue_size);

  // SIM ANNEAL
  simAnneal();
}

void SimAnneal::runSimCUDA()
{
  cudaFree(0);  // does nothing, absorbs the startup delay when profiling

  // initialize variables & perform pre-calculations
  float kT = 300*constants::Kb; // kT = Boltzmann constant (eV/K) * 300K

  const int max_handle_init_threads = 4;

  const int max_blocks_per_stream = 8;
  const int max_threads_per_block = 32;
  int num_streams = ceil((float)num_threads / (max_blocks_per_stream * max_threads_per_block));
  int num_blocks = num_streams > 1 ? max_blocks_per_stream : ceil((float)num_threads / max_threads_per_block);
  int threads_per_block = num_blocks > 1 ? max_threads_per_block : num_threads;
  int threads_per_stream = num_blocks * threads_per_block;
  
  float *d_v_ext, *d_db_locs;//, *d_v_ij;
  cudaMallocManaged(&d_v_ext, n_dbs*sizeof(float));
  cudaMallocManaged(&d_db_locs, 2*n_dbs*sizeof(float));

  // one set of n_out per stream
  std::vector<float*> return_arrays;
  for (int i=0; i<num_streams; i++) {
    float *n_out_set;
    cudaMallocManaged(&n_out_set, threads_per_stream * result_queue_size * n_dbs * sizeof(float));
    return_arrays.push_back(n_out_set);
  }

  for (int i=0; i<n_dbs; i++) {
    d_v_ext[i] = v_ext[i];
    d_db_locs[IDX2C(i,0,n_dbs)] = db_locs[i].first * db_distance_scale;   // x
    d_db_locs[IDX2C(i,1,n_dbs)] = db_locs[i].second * db_distance_scale;  // y
  }
  cudaDeviceSynchronize();

  //cudaStream_t streams[num_streams];
  cudaStream_t *streams = (cudaStream_t*)malloc(num_streams*sizeof(cudaStream_t));

  std::cout << "Initializing cublas handle and SimAnneal constants" << std::endl;
  int handle_init_threads = num_streams > max_handle_init_threads ? max_handle_init_threads : num_streams;
  ::initCublasHandles<<<1,handle_init_threads>>>(num_streams);  // initializing too many cublas handles at once could crash the program
  ::initDeviceVars<<<1,1>>>(num_streams, n_dbs, debye_length, mu, kT, kT0, kT_step, v_freeze_step, t_max, d_v_ext, d_db_locs);
  cudaPeekAtLastError();
  cudaDeviceSynchronize();

  for (int i=0; i < num_streams; i++) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    std::cout << "Invoking CUDA SimAnneal <<<" << num_blocks << "," << threads_per_block << ",0,streams[" << i << "]>>>" << std::endl;
    ::simAnnealParallel<<<num_blocks,threads_per_block,0,streams[i]>>>(i, result_queue_size, return_arrays[i]);
  }
  cudaPeekAtLastError();
  cudaDeviceSynchronize();

  for (int i=0; i < num_streams; i++) {
    cudaStreamDestroy(streams[i]);
  }
  cudaDeviceSynchronize();

  std::cout << "destroying cublas handle" << std::endl;
  ::destroyCublasHandles<<<1,num_streams>>>(num_streams);
  ::cleanUpDeviceVars<<<1,1>>>(num_streams);
  cudaPeekAtLastError();
  //gpuErrChk(cudaDeviceSynchronize());

  // TODO move results to a form understood by SiQADConn
  // return all sets of n_out
  for (float *n_out_set : return_arrays) {
    // loop through each set
    for (int i=0; i<result_queue_size; i++) {
      ublas::vector<int> charges;
      charges.resize(n_dbs);

      // loop through DBs within each set
      for (int j=0; j<n_dbs; j++)
        charges[j] = static_cast<int>(n_out_set[i*n_dbs+j]);

      db_charges.push_back(charges);
      config_energies.push_back(systemEnergy(charges));
    }
    gpuErrChk(cudaFree(n_out_set));
  }
  
  cudaDeviceSynchronize();

  // clean up
  cudaFree(d_v_ext);
  cudaFree(d_db_locs);

  cudaDeviceReset();

  // export
  writeStore(this, threadId);
}


void SimAnneal::simAnneal()
{
  // Vars
  boost::numeric::ublas::vector<int> dn(n_dbs); // change of occupation for population update
  int from_occ_ind, to_occ_ind; // hopping from n[occ[from_ind]]
  int from_ind, to_ind;         // hopping from n[from_ind] to n[to_ind]
  int hop_attempts;

  //n_best.resize(n.size());      //Variables used in restarting. uncomment
  //firstBest = false;            //these two for restarts.

  E_sys = systemEnergy();
  //E_best = E_sys;         // initializing the best system energy with the initial energy
  //n_best = n;             //initializing the best electron configuration with the initial electron config.
  v_local = v_ext - ublas::prod(v_ij, n);

  //steadyPopCount = 0;           //Variable for restarting. Uncomment when restarting.

  
  /*
  // arrays for CUDA code
  float *n_arr, *v_ext_arr, *v_ij_arr, *cuda_v, *dn_arr, *v_local_arr;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&n_arr, n_dbs*sizeof(float));
  cudaMallocManaged(&v_local_arr, n_dbs*sizeof(float));
  cudaMallocManaged(&v_ext_arr, n_dbs*sizeof(float));
  cudaMallocManaged(&v_ij_arr, n_dbs*n_dbs*sizeof(float));
  cudaMallocManaged(&cuda_v, sizeof(float));
  cudaMallocManaged(&dn_arr, n_dbs*sizeof(float));

  ::initCublasHandle<<<1, 1>>>();
  */


  // Run simulated annealing for predetermined time steps
  while(t < t_max) {

    // Population
    dn = genPopDelta();

    bool pop_changed = false;
    for (unsigned i=0; i<dn.size(); i++) {
      if (dn[i] != 0) {
        pop_changed = true;
        break;
      }
    }

/*        //Used in Restarting. Uncomment if restarting.
    if(pop_changed){
      steadyPopCount = 0;
    }
    else{
      steadyPopCount++;
    }
*/

    if (pop_changed) {
      n += dn;
      E_sys += -1 * ublas::inner_prod(v_local, dn) + totalCoulombPotential(dn);
      v_local -= ublas::prod(v_ij, dn);



      /*
      // NOTE CUDA test
      float c_ver = totalCoulombPotential(dn);
      float c_E_sys_del = -1 * ublas::inner_prod(v_local, dn) + totalCoulombPotential(dn);
      //float c_E_sys_del = -1 * ublas::inner_prod(v_local, dn);

      // copy vector data to array
      for (int i=0; i<n_dbs; i++) {
        dn_arr[i] = dn[i];
        n_arr[i] = n[i];
        v_ext_arr[i] = v_ext[i];
        v_local_arr[i] = v_local[i];
        for (int j=0; j<n_dbs; j++) {
          v_ij_arr[IDX2C(i,j,n_dbs)] = v_ij(i,j);
        }
      }
      *cuda_v = 0;
      ::totalCoulombPotential<<<1, 1>>>(n_dbs, dn_arr, v_ij_arr, cuda_v);
      cudaDeviceSynchronize();

      std::cout << "total coulomb potential c++ :   " << c_ver << std::endl;
      std::cout << "total coulomb potential cuda:   " << *cuda_v << std::endl;
      std::cout << std::endl;

      *cuda_v = 0;
      ::populationChangeEnergyDelta<<<1, 1>>>(n_dbs, dn_arr, v_ij_arr, v_local_arr, cuda_v);
      cudaDeviceSynchronize();

      std::cout << "pop change c++ : " << c_E_sys_del << std::endl;
      std::cout << "pop change cuda: " << *cuda_v << std::endl;
      std::cout << std::endl;
      */
    }


    // Occupation list update
    int occ_ind=0, unocc_ind=n_dbs-1;
    for (int db_ind=0; db_ind<n_dbs; db_ind++) {
      if (n[db_ind])
        occ[occ_ind++] = db_ind;
      else
        occ[unocc_ind--] = db_ind;
    }
    n_elec = occ_ind;


    // Hopping
    hop_attempts = 0;
    if (n_elec != 0) {
      while (hop_attempts < (n_dbs-n_elec)*5) {
        from_occ_ind = getRandOccInd(1);
        to_occ_ind = getRandOccInd(0);
        from_ind = occ[from_occ_ind];
        to_ind = occ[to_occ_ind];

        float E_del = hopEnergyDelta(from_ind, to_ind);
        if (acceptHop(E_del)) {
          performHop(from_ind, to_ind);
          occ[from_occ_ind] = to_ind;
          occ[to_occ_ind] = from_ind;
          // calculate energy difference
          E_sys += E_del;
          ublas::matrix_column<ublas::matrix<float>> v_i (v_ij, from_ind);
          ublas::matrix_column<ublas::matrix<float>> v_j (v_ij, to_ind);
          v_local += v_i - v_j;
        }
        hop_attempts++;
      }
    }

    // push back the new arrangement
    db_charges.push_back(n);
    config_energies.push_back(E_sys);

    // perform time-step if not pre-annealing
    timeStep();
  }

  /*
  // copy vector data to array
  for (int i=0; i<n_dbs; i++) {
    n_arr[i] = n[i];
    v_ext_arr[i] = v_ext[i];
    std::cout << "n_arr[" << i << "] = " << n_arr[i] << std::endl;
    std::cout << "v_ext_arr[" << i << "] = " << v_ext_arr[i] << std::endl;
    for (int j=0; j<n_dbs; j++) {
      v_ij_arr[i*n_dbs + j] = v_ij(i,j);
      std::cout << "v_ij_arr[" << i*n_dbs+j << "] = " << v_ij_arr[i*n_dbs+j] << std::endl;
    }
  }

  //*cuda_v = 0;
  //::systemEnergy<<<1, 1>>>(cuda_v, n_dbs, n_arr, v_ext_arr, v_ij_arr);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  std::cout << "Host systemEnergy()=" << systemEnergy() << std::endl;
  //std::cout << "CUDA systemEnergy()=" << *cuda_v << std::endl;

  *cuda_v = 0;
  //::systemEnergyCublas<<<1, 1>>>(n_dbs, n_arr, v_ext_arr, v_ij_arr, cuda_v);
  ::systemEnergy<<<1, 1>>>(n_dbs, n_arr, v_ext_arr, v_ij_arr, cuda_v);
  cudaDeviceSynchronize();
  std::cout << "cublas systemEnergy=" << *cuda_v << std::endl;
  
  // Free memory
  cudaFree(n_arr);
  cudaFree(v_ext_arr);
  cudaFree(v_ij_arr);
  cudaFree(cuda_v);
  cudaFree(dn_arr);
  ::destroyCublasHandle<<<1, 1>>>();*/

  writeStore(this, threadId);
}










ublas::vector<int> SimAnneal::genPopDelta()
{
  ublas::vector<int> dn(n_dbs);
  for (unsigned i=0; i<n.size(); i++) {
    //float prob = 1. / ( 1 + exp( ((2*n[i]-1)*v_local[i] + v_freeze) / kT ) );
    float prob = 1. / ( 1 + exp( ((2*n[i]-1)*(v_local[i] + mu) + v_freeze) / kT ) );
    std::cout << "n[i]=" << n[i] << ", v_local[i]=" << v_local[i] 
      << ", mu=" << mu << ", v_freeze=" << v_freeze << ", kT=" << kT 
      << ", prob=" << prob << std::endl;
    dn[i] = evalProb(prob) ? 1 - 2*n[i] : 0;
  }
  return dn;
}

void SimAnneal::performHop(int from_ind, int to_ind)
{
  n[from_ind] = 0;
  n[to_ind] = 1;
}


void SimAnneal::timeStep()
{
  t++;
  kT = kT0 + (kT - kT0) * kT_step;
  v_freeze = t * v_freeze_step;

/*
  //simAnneal restarts
  if(!firstBest){
    firstBest = true;
    E_best = E_sys;
  }

  if(steadyPopCount > STEADY_THREASHOLD && E_sys < E_best){
    E_best = E_sys;
    n_best = n;
  }


  if( steadyPopCount > STEADY_THREASHOLD && (E_sys > 1.1*E_best || evalProb(0)) && t < 0.99*t_max){
    //t-=0.05*t_max;
    E_sys = E_best;
    n = n_best;
    std::cout << "******************RESTART******************" << std::endl;
  }
*/
}

// ACCEPTANCE FUNCTIONS

// acceptance function for hopping
bool SimAnneal::acceptHop(float v_diff)
{
  if (v_diff < 0)
    return true;

  // some acceptance function, acceptance probability falls off exponentially
  float prob = exp(-v_diff/kT);

  return evalProb(prob);
}


// takes a probability and generates true/false accordingly
bool SimAnneal::evalProb(float prob)
{
  //float generated_num = dis01(rng);
  boost::variate_generator<boost::random::mt19937&, boost::random::uniform_real_distribution<float>> rnd_gen(rng, dis01);

  return prob >= rnd_gen();
}





// ACCESSORS


int SimAnneal::getRandOccInd(int charge)
{
  int min,max;
  if (charge) {
    min = 0;
    max = n_elec-1;
  } else {
    min = n_elec;
    max = n_dbs-1;
  }
  boost::random::uniform_int_distribution<int> dis(min,max);
  return dis(rng);
}





// PHYS CALCULATION


float SimAnneal::systemEnergy()
{
  assert(n_dbs > 0);
  float v = 0;
  for(int i=0; i<n_dbs; i++) {
    //v -= mu + v_ext[i] * n[i];
    v -= v_ext[i] * n[i];
    for(int j=i+1; j<n_dbs; j++)
      v += v_ij(i,j) * n[i] * n[j];
  }
  return v;
}


float SimAnneal::systemEnergy(ublas::vector<int> config)
{
  assert(n_dbs > 0);
  float v = 0;
  for(int i=0; i<n_dbs; i++) {
    //v -= mu + v_ext[i] * n[i];
    v -= v_ext[i] * config[i];
    for(int j=i+1; j<n_dbs; j++)
      v += v_ij(i,j) * config[i] * config[j];
  }
  return v;
}


float SimAnneal::distance(const float &x1, const float &y1, const float &x2, const float &y2)
{
  return sqrt(pow(x1-x2, 2.0) + pow(y1-y2, 2.0));
}


float SimAnneal::totalCoulombPotential(ublas::vector<int> config)
{
  return 0.5 * ublas::inner_prod(config, ublas::prod(v_ij, config));
}


float SimAnneal::interElecPotential(const float &r)
{
  //return exp(-r/debye_length) / r;
  return constants::Q0 * Kc * erf(r/constants::ERFDB) * exp(-r/debye_length) / r;
}


float SimAnneal::hopEnergyDelta(const int &i, const int &j)
{
  return v_local[i] - v_local[j] - v_ij(i,j);
}
