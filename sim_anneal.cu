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

__device__ cublasHandle_t cb_hdl;
__device__ float mu, kT0, kT_step, v_freeze_step;
__device__ float *v_ij;

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


__global__ void simAnnealAlg(int n_dbs, float *v_ext, int t_max, float kT_init)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tId != 0) return;

  // cycle tracking
  int t=0;                      // current anneal cycle

  // population related vars
  int n_elec;                   // number of electrons in the system
  bool *pop_changed;            // indicate whether population has changed during this cycle
  float *v_freeze;              // current freeze out voltage (population)
  float *kT;                    // current temperature (population)
  cudaMalloc(&pop_changed, sizeof(bool));
  cudaMalloc(&v_freeze, sizeof(float));
  cudaMalloc(&kT, sizeof(float));
  *v_freeze = 0;
  *kT = kT_init;

  // hop related vars
  const int hop_threads=32;     // number of threads to run hop attempts in parallel
  int *occ;                     // first n_elec elements are indices of occupied sites in the n array; the rest are unoccupied indices.
  float *n_ph;                  // pointer of size n_dbs*hop_threads*sizeof(float) that stores the electron config from all parallel hop paths
  float *v_local_ph;            // pointer of size n_dbs*hop_threads*sizeof(float) that stores the v_local of all parallel hop paths
  float *E_del_ph;              // energy delta from parallel hop attempts
  cudaMalloc(&occ, n_dbs*sizeof(int));
  cudaMalloc(&n_ph, n_dbs*hop_threads*sizeof(float));
  cudaMalloc(&v_local_ph, n_dbs*hop_threads*sizeof(float));
  cudaMalloc(&E_del_ph, hop_threads*sizeof(float));

  float *n;                     // current occupation of DB sites
  float *dn;                    // change of occupation for population update
  float *v_local;               // local potential at each site
  cudaMalloc(&n, n_dbs*sizeof(float));
  cudaMalloc(&dn, n_dbs*sizeof(float));
  cudaMalloc(&v_local, n_dbs*sizeof(float));

  for (int i=0; i<n_dbs; i++) {
    n[i] = 0.;
    dn[i] = 0.;
    v_local[i] = 0.;
  }

  //print2DArrayFloat("v_ij", v_ij, n_dbs);

  // initialize system energy and local energy
  printf("Initializing system energy and local energy\n");
  float E_sys;
  float E_del;
  systemEnergy(n_dbs, n, v_ext, &E_sys);
  initVLocal(n_dbs, n, v_ext, v_local);

  printf("\n***Beginning simanneal***\n\n");
  while (t < t_max) {
    // Population
    *pop_changed = false;
    genPopulationDelta<<<1,n_dbs>>>(n_dbs, n, v_local, v_freeze, kT, dn, pop_changed);
    cudaDeviceSynchronize();
    __syncthreads();
    if (*pop_changed) {
      // n + dn
      float alpha=1;
      cublasCheckErrors(cublasSaxpy(cb_hdl, n_dbs, &alpha, dn, 1, n, 1));
      __syncthreads();

      // E_sys += energy delta from population change
      populationChangeEnergyDelta(n_dbs, dn, v_local, &E_del);
      __syncthreads();
      E_sys += E_del;

      // v_local = - prod(v_ij, dn) + v_local 
      alpha=-1;
      float beta=1;
      cublasCheckErrors(cublasSgemv(cb_hdl, CUBLAS_OP_N, n_dbs, n_dbs, &alpha, v_ij, n_dbs, dn, 1,
          &beta, v_local, 1));
      __syncthreads();

      //print1DArrayFloat("Population changed, dn", dn, n_dbs);
      //print1DArrayFloat("v_local", v_local, n_dbs);
      //print1DArrayFloat("n", n, n_dbs);

      // occupation list update
      int occ_ind=0, unocc_ind=n_dbs-1;
      for (int db_ind=0; db_ind<n_dbs; db_ind++) {
        if (n[db_ind] == 1.0)
          occ[occ_ind++] = db_ind;
        else
          occ[unocc_ind--] = db_ind;
      }
      n_elec = occ_ind;
      //print1DArrayInt("occ", occ, n_dbs);
    }
    //printf("\n");

    
    // Hopping
    // TODO currently arbitrary hop attempts, make configurable
    if (n_elec != 0 && n_elec != n_dbs) {
      int max_hops = (n_dbs-n_elec)*5;  
      parallelHops<<<1,hop_threads>>>(n_dbs, n_elec, max_hops, kT,
          n, n_ph, v_local, v_local_ph, E_del_ph);
      cudaDeviceSynchronize();

      // find best hop path amongst the attempted
      float best_E_del=E_del_ph[0];
      float best_ind=0;
      for (int i=1; i<hop_threads; i++) {
        //printf("i=%d, n_dbs=%d", i, n_dbs);
        if (E_del_ph[i] < best_E_del) {
          best_E_del = E_del_ph[i];
          best_ind=i;
        }
      }
      
      // assign best output to n and v_local
      E_sys += best_E_del;
      int i_offset = best_ind * n_dbs;
      for (int i=0; i<n_dbs; i++) {
        n[i] = n_ph[i+i_offset];
        v_local[i] = v_local_ph[i+i_offset];
      }
      //printf("Best index=%d, i_offset=%d\n", best_ind, i_offset);
      //printf("Best E_del=%f\n", best_E_del);
      //print1DArrayFloat("Best n", n, n_dbs);
      //print1DArrayFloat("Best v_local", v_local, n_dbs);
      //printf("\n");
    }
    //printf("\n");

    // TODO store new arrangement

    // time step
    timeStep(&t, kT, v_freeze);
    __syncthreads();
    //printf("Cycle: %d, ending energy: %f\n\n", t, E_sys);
  }

  /*print1DArrayFloat("Final n\n", n, n_dbs);
  printf("Ending energy (delta): %f\n", E_sys);

  systemEnergy(n_dbs, n, v_ext, &E_sys);
  printf("Ending energy (actua): %f\n", E_sys);*/

  free(pop_changed);
  free(v_freeze);
  free(kT);
  free(occ);
  free(n_ph);
  free(v_local_ph);
  free(E_del_ph);
  free(n);
  free(dn);
  free(v_local);
  //cublasCheckErrors(cublasDestroy_v2(cb_hdl_2));
}

__global__ void parallelHops(int n_dbs, int n_elec, int max_hops, float *kT, 
    float *n_start, float *n_ph, float *v_local_start, float *v_local_ph, float *E_del_ph)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);

  // thread specific offsets
  int i_offset = tId*n_dbs;                   // index offset for n_ph and v_local_ph

  // hop variables
  int hop_attempts=0;                         // number of hops attempted
  int from_occ_ind, to_occ_ind;               // from & to indices in the occ array, where the first n_elec elements are indices of occupied sites in n or n_start
  int from_ind, to_ind;                       // hop from & to indicated sites
  int n_empty = n_dbs - n_elec;               // number of empty sites
  bool accept_hop;
  float E_del;                                // delta E of an attempted hop

  // copy elements from n_start to n, v_local_start to v_local
  //float *n=&n_ph[i_offset];                   // electron configurations after all successful hops
  //float *v_local=&v_local_ph[i_offset];       // accumulated changes to v_local of all successful hops in this path
  //float *E_del_total=&E_del_ph[tId];          // accumulated delta E of all successful hops in this path
  //*E_del_total = 0;
  //float *n, *v_local;
  float *v_local;
  int *occ;
  //float E_del_accum=0;
  //cudaMalloc(&n, n_dbs*sizeof(float));
  cudaMalloc(&v_local, n_dbs*sizeof(float));
  cudaMalloc(&occ, n_dbs*sizeof(int));
  E_del_ph[tId]=0;
  for (int i=0; i<n_dbs; i++) {
    //n[i] = n_start[i];
    n_ph[i+i_offset] = n_start[i];
    v_local[i] = v_local_start[i];
  }
  int occ_ind=0, unocc_ind=n_dbs-1;
  for (int db_ind=0; db_ind<n_dbs; db_ind++) {
    if (n_start[db_ind] == 1.0)
      occ[occ_ind++] = db_ind;
    else
      occ[unocc_ind--] = db_ind;
  }
  n_elec = occ_ind;
  //print1DArrayFloat("\ncopy_n", n, n_dbs);
  //print1DArrayFloat("copy_v_local", v_local, n_dbs);

  // RNG
  curandState curand_state;
  curand_init((unsigned long long)clock() + tId, 0, 0, &curand_state);

  // hop loop
  while (hop_attempts < max_hops) {
    randInt(&curand_state, n_elec, &from_occ_ind);
    randInt(&curand_state, n_empty, &to_occ_ind);
    to_occ_ind += n_elec;
    from_ind = occ[from_occ_ind];
    to_ind = occ[to_occ_ind];

    //printf("from_occ_ind=%d, to_occ_ind=%d\n", from_occ_ind, to_occ_ind);

    hopEnergyDelta(from_ind, to_ind, n_dbs, v_local, &E_del);
    //E_del = v_local_ph[from_ind+i_offset] - v_local_ph[to_ind+i_offset] - v_ij[IDX2C(from_ind,to_ind,n_dbs)];
    //E_del = v_local[from_ind] - v_local[to_ind] - v_ij[IDX2C(from_ind,to_ind,n_dbs)];
    //__syncthreads();
    acceptHop(&curand_state, &E_del, kT, &accept_hop);
    //__syncthreads();

    //printf("Attempting hop from sites %d to %d with E_del=%f...\n", from_ind, to_ind, E_del);
    if (accept_hop) {
      // update relevant arrays to keep track of hops
      n_ph[from_ind+i_offset] = 0.;
      n_ph[to_ind+i_offset] = 1.;
      occ[from_occ_ind] = to_ind;
      occ[to_occ_ind] = from_ind;
      //printf("Post hop:\nn[%d]=%f, n[%d]=%f\nocc[%d]=%d,occ[%d]=%d\n", from_ind, n[from_ind], to_ind, n[to_ind], from_occ_ind, occ[from_occ_ind], to_occ_ind, occ[to_occ_ind]);

      // update energy
      E_del_ph[tId] += E_del;
      //E_del_accum += E_del;
      //updateVLocal<<<1,n_dbs>>>(from_ind, to_ind, n_dbs, v_local);
      updateVLocal(from_ind, to_ind, n_dbs, v_local);
      cudaDeviceSynchronize();
      /*for (int i=0; i<n_dbs; i++) {
        v_local[i] += v_ij[IDX2C(i, from_ind, n_dbs)] - v_ij[IDX2C(i, to_ind, n_dbs)];
      }*/
      //__syncthreads();
      //print1DArrayFloat("Accepted. New v_local=", v_local, n_dbs);
      //print1DArrayFloat("New n=", n, n_dbs);
    }
    hop_attempts++;
  }
  for (int i=0; i<n_dbs; i++) {
    v_local_ph[i+i_offset] = v_local[i];
    //n_ph[i+i_offset] = n[i];
    //E_del_ph[tId] = E_del_accum;
  }

  //printf("Hop result for tId=%d:\n", tId);
  //print1DArrayFloat("n", n, n_dbs);
  //free(n);
  free(v_local);
  free(occ);
}

__global__ void initDeviceVars(float n_dbs, float debye_length, float mu_in, 
    float kT0_in, float kT_step_in, float v_freeze_step_in, float *db_locs)
{
  // create cublas handle
  printf("Initializing cublas handle\n");
  cublasCreate_v2(&cb_hdl);

  // assign simple variables
  printf("Assigning variables\n");
  mu = mu_in;
  kT0 = kT0_in;
  kT_step = kT_step_in;
  v_freeze_step = v_freeze_step_in;

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

__global__ void cleanUpDeviceVars()
{
  cublasCheckErrors(cublasDestroy_v2(cb_hdl));
  free(v_ij);
}

__device__ void initVLocal(int n_dbs, float *n, float *v_ext, float *v_local)
{
  // v_local = v_ext - dot(v_ij, n)

  // dot(v_ij, n)
  float alpha=1;
  float beta=0;
  cublasCheckErrors(cublasSgemv(cb_hdl, CUBLAS_OP_N, n_dbs, n_dbs, &alpha, v_ij, n_dbs, n, 1, 
      &beta, v_local, 1));
  __syncthreads();

  // v_ext - above
  alpha=-1;
  cublasCheckErrors(cublasSaxpy(cb_hdl, n_dbs, &alpha, v_local, 1, v_ext, 1));
  __syncthreads();
}

__device__ void updateVLocal(int from_ind, int to_ind, int n_dbs, float *v_local)
{
  //v_local += v_ij(from-th col) - v_ij(to-th col);

  //int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  //int stride = blockDim.x * gridDim.x;

  //for (int i=tId; i<n_dbs; i+=stride) {
  for (int i=0; i<n_dbs; i++) {
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
    float prob = 1. / ( 1. + expf( ((2.*n[i]-1.)*(v_local[i]+mu) + *v_freeze ) / *kT ));
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

__device__ void systemEnergy(int n_dbs, float *n, float *v_ext, float *output)
{
  // TODO might be able to merge this function with population change energy delta with the similarity
  float *coulomb_v;
  cudaMalloc(&coulomb_v, sizeof(float));
  totalCoulombPotential(n_dbs, n, coulomb_v);
  __syncthreads();
  cublasCheckErrors(cublasSdot(cb_hdl, n_dbs, v_ext, 1, n, 1, output));
  __syncthreads();
  *output *= -1;
  *output += *coulomb_v;
  free(coulomb_v);
}

// Energy change from population change.
__device__ void populationChangeEnergyDelta(int n_dbs, float *dn, float *v_local, float *output)
{
  // delta E = -1 * dot(v_local, dn) + dn^T * V_ij * dn
  float *coulomb_v = (float*)malloc(sizeof(float));
  totalCoulombPotential(n_dbs, dn, coulomb_v);
  __syncthreads();
  cublasCheckErrors(cublasSdot(cb_hdl, n_dbs, v_local, 1, dn, 1, output));
  __syncthreads();
  *output *= -1;
  *output += *coulomb_v;
  free(coulomb_v);
}

// Total potential from Coulombic repulsion in the system.
__device__ void totalCoulombPotential(int n_dbs, float *n, float *v)
{
  float alpha=0.5;
  float beta=0;
  float *v_temp;
  cudaMalloc(&v_temp, n_dbs*sizeof(float));
  cublasCheckErrors(cublasSgemv(cb_hdl, CUBLAS_OP_N, n_dbs, n_dbs, &alpha, v_ij, n_dbs, n, 1, &beta, v_temp, 1));
  __syncthreads();
  cublasSdot(cb_hdl, n_dbs, n, 1, v_temp, 1, v);
  __syncthreads();
  free(v_temp);
}

__device__ void acceptHop(curandState *curand_state, float *v_diff, float *kT, bool *accept)
{
  if (*v_diff <= 0.) {
    *accept = true;
  } else {
    float prob = expf( -(*v_diff) / (*kT) );
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

  db_charges.resize(result_queue_size);
  n.resize(n_dbs);
  occ.resize(n_dbs);

  config_energies.resize(result_queue_size);

  // SIM ANNEAL
  simAnneal();
}

void SimAnneal::runSimCUDA()
{
  cudaFree(0);  // does nothing, absorbs the startup delay when profiling

  // initialize variables & perform pre-calculations
  float kT = 300*constants::Kb; // kT = Boltzmann constant (eV/K) * 300K
  
  //float *v_ext_arr, *db_locs_arr;
  float *d_v_ext, *d_db_locs;//, *d_v_ij;
  gpuErrChk(cudaMallocManaged(&d_v_ext, n_dbs*sizeof(float)));
  gpuErrChk(cudaMallocManaged(&d_db_locs, 2*n_dbs*sizeof(float)));

  for (int i=0; i<n_dbs; i++) {
    d_v_ext[i] = v_ext[i];
    d_db_locs[IDX2C(i,0,n_dbs)] = db_locs[i].first * db_distance_scale;   // x
    d_db_locs[IDX2C(i,1,n_dbs)] = db_locs[i].second * db_distance_scale;  // y
  }
  gpuErrChk(cudaDeviceSynchronize());

  //cudaStream_t streams[num_streams];
  cudaStream_t *streams = (cudaStream_t*)malloc(num_threads*sizeof(cudaStream_t));

  std::cout << "initializing CUDA SimAnneal constants" << std::endl;
  ::initDeviceVars<<<1,1>>>(n_dbs, debye_length, mu, kT0, kT_step, v_freeze_step, d_db_locs);
  gpuErrChk(cudaPeekAtLastError());
  gpuErrChk(cudaDeviceSynchronize());

  for (int i=0; i < num_threads; i++) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    std::cout << "invoking CUDA SimAnneal..." << std::endl;
    ::simAnnealAlg<<<1,1,0,streams[i]>>>(n_dbs, d_v_ext, t_max, kT);
  }
  gpuErrChk(cudaPeekAtLastError());
  gpuErrChk(cudaDeviceSynchronize());

  for (int i=0; i < num_threads; i++) {
    cudaStreamDestroy(streams[i]);
  }
  gpuErrChk(cudaDeviceSynchronize());

  std::cout << "destroying cublas handle" << std::endl;
  ::cleanUpDeviceVars<<<1,1>>>();
  gpuErrChk(cudaPeekAtLastError());
  gpuErrChk(cudaDeviceSynchronize());

  // TODO move results to a form understood by SiQADConn
  
  // clean up
  gpuErrChk(cudaFree(d_v_ext));
  gpuErrChk(cudaFree(d_db_locs));

  gpuErrChk(cudaDeviceReset());
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
