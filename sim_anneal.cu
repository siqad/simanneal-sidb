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

// CUDA error checking
#define cudaCheckErrors(msg) \
  if (cudaGetLastError() != cudaSuccess) { \
    std::cerr << "CUDA error: " << msg << "(" << cudaGetErrorString(cudaGetLastError()) \
      << " at " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
    exit(1); \
  }

// cuBLAS error checking
#define cublasCheckErrors(cublas_status) \
  if (cublas_status != CUBLAS_STATUS_SUCCESS) { \
    std::cerr << "Fatal cuBLAS error: " << (int)(cublas_status) << "(at " << \
      __FILE__ << ":" << __LINE__ << ")" << std::endl; \
    exit(1); \
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


__global__ void simAnnealAlg(int n_dbs, float *v_ext, float *v_ij, int t_max, float kT_init)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tId != 0) return;

  // cycle tracking
  int t=0;                      // current anneal cycle

  // population related vars
  int n_elec;                   // number of electrons in the system
  bool pop_changed;             // indicate whether population has changed during this cycle
  float v_freeze=0;             // current freeze out voltage (population)
  float kT=kT_init;             // current temperature (population)

  // hop related vars
  int *occ;                     // first n_elec elements are indices of occupied sites in the n array; the rest are unoccupied indices.
  int from_occ_ind, to_occ_ind; // hopping from n[occ[from_ind]]
  int from_ind, to_ind;         // hopping from n[from_ind] to n[to_ind]
  int hop_attempts;             // number of hop attempts so far within a SimAnneal cycle
  float *v_i_temp, *v_j_temp;   // temporary storage of columns i and j from v_ij
  occ = (int*)malloc(n_dbs*sizeof(int));
  v_i_temp = (float*)malloc(n_dbs*sizeof(float));
  v_j_temp = (float*)malloc(n_dbs*sizeof(float));

  float *n;                     // current occupation of DB sites
  float *dn;                    // change of occupation for population update
  float *v_local;               // local potential at each site
  n = (float*)malloc(n_dbs*sizeof(float));
  dn = (float*)malloc(n_dbs*sizeof(float));
  v_local = (float*)malloc(n_dbs*sizeof(float));

  for (int i=0; i<n_dbs; i++) {
    n[i] = 0.;
    dn[i] = 0.;
    v_local[i] = 0.;
  }

  // initialize system energy and local energy
  printf("Initializing system energy and local energy\n");
  float E_sys;
  float E_del;
  systemEnergy(n_dbs, n, v_ext, v_ij, &E_sys);
  initVLocal(n_dbs, n, v_ext, v_ij, v_local);

  printf("\n***Beginning simanneal***\n\n");
  // TODO learn how to do curand state offsets
  while (t < t_max) {
    // Population
    pop_changed = false;
    genPopulationDelta(n_dbs, n, v_local, &v_freeze, &kT, dn, &pop_changed);
    __syncthreads();
    printf("sync up\n");
    if (pop_changed) {
      // n + dn
      float alpha=1;
      cublasSaxpy(cb_hdl, n_dbs, &alpha, dn, 1, n, 1);
      __syncthreads();

      // E_sys += energy delta from population change
      populationChangeEnergyDelta(n_dbs, dn, v_ij, v_local, &E_del);
      __syncthreads();
      E_sys += E_del;

      // v_local = - prod(v_ij, dn) + v_local 
      alpha=-1;
      float beta=1;
      cublasSgemv(cb_hdl, CUBLAS_OP_N, n_dbs, n_dbs, &alpha, v_ij, n_dbs, dn, 1,
          &beta, v_local, 1);
      __syncthreads();

      /*print1DArrayFloat("Population changed, dn", dn, n_dbs);
      print1DArrayFloat("v_local", v_local, n_dbs);
      print1DArrayFloat("n", n, n_dbs);*/

      // occupation list update
      // TODO parallelize
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
    // TODO try to get multiple attempts to go in parallel, then pick the best
    // TODO probably want to move all of the variable declarations to the top
    // TODO currently arbitrary hop attempts, make configurable
    /*if (n_elec != 0) {
      //printf("Hopping\n\n");
      hop_attempts = 0;
      int max_hop_attempts = (n_dbs-n_elec)*5;  
      while (hop_attempts < max_hop_attempts) {
        // TODO make sure that all indices have a chance of being picked.
        randInt(n_elec, &from_occ_ind);
        randInt(n_dbs-n_elec, &to_occ_ind);
        to_occ_ind += n_elec;
        from_ind = occ[from_occ_ind];
        to_ind = occ[to_occ_ind];

        bool accept_hop;
        hopEnergyDelta(from_ind, to_ind, n_dbs, v_local, v_ij, &E_del);
        __syncthreads();
        acceptHop(&E_del, &kT, &accept_hop);
        __syncthreads();
        //printf("Attempting hop from sites %d to %d with E_del=%f...\n", from_ind, to_ind, E_del);
        if (accept_hop) {
          // TODO maybe fold the following into one simple function
          n[from_ind] = 0.;
          n[to_ind] = 1.;
          occ[from_occ_ind] = to_ind;
          occ[to_occ_ind] = from_ind;

          // update energy
          E_sys += E_del;
          updateVLocal(from_ind, to_ind, n_dbs, v_ij, v_local);
          __syncthreads();

          //print1DArrayFloat("Accepted. New v_local=", v_local, n_dbs);
        }
        hop_attempts++;
      }
    }*/
    //printf("\n");

    // TODO store new arrangement

    // time step
    timeStep(&t, &kT, &v_freeze);
    __syncthreads();
    printf("Cycle: %d, ending energy: %f\n\n", t, E_sys);
  }

  print1DArrayFloat("Final n\n", n, n_dbs);
  printf("Ending energy (delta): %f\n", E_sys);

  systemEnergy(n_dbs, n, v_ext, v_ij, &E_sys);
  printf("Ending energy (actua): %f\n", E_sys);

  free(occ);
  free(v_i_temp);
  free(v_j_temp);
  free(n);
  free(dn);
  free(v_local);
}


__global__ void initCublasHandle()
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tId == 0)
    cublasCreate(&cb_hdl);
}

__global__ void destroyCublasHandle()
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tId == 0)
    cublasDestroy(cb_hdl);
}

__global__ void initSimAnnealConsts(float mu_in, float kT0_in, 
    float kT_step_in, float v_freeze_step_in)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tId != 0) return;

  mu = mu_in;
  kT0 = kT0_in;
  kT_step = kT_step_in;
  v_freeze_step = v_freeze_step_in;
}

__device__ void initVLocal(int n_dbs, float *n, float *v_ext, float *v_ij, float *v_local)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tId != 0) return;
  // v_local = v_ext - dot(v_ij, n)

  // dot(v_ij, n)
  float alpha=1;
  float beta=0;
  cublasSgemv(cb_hdl, CUBLAS_OP_N, n_dbs, n_dbs, &alpha, v_ij, n_dbs, n, 1, 
      &beta, v_local, 1);
  __syncthreads();

  // v_ext - above
  alpha=-1;
  cublasSaxpy(cb_hdl, n_dbs, &alpha, v_local, 1, v_ext, 1);
  __syncthreads();
}

__device__ void updateVLocal(int from_ind, int to_ind, int n_dbs, float *v_ij, float *v_local)
{
  //v_local += v_ij(from-th col) - v_ij(to-th col);
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  int stride = blockDim.x * gridDim.x;

  for (int i=tId; i<n_dbs; i+=stride) {
    v_local[i] += v_ij[IDX2C(i, from_ind, n_dbs)] - v_ij[IDX2C(i, to_ind, n_dbs)];
  }
}

__device__ void genPopulationDelta(int n_dbs, float *n, float *v_local, 
    float *v_freeze, float *kT, float *dn, 
    bool *pop_changed)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  //int stride = blockDim.x * gridDim.x;
  //if (tId != 0) return;

  curandState curand_state;
  curand_init((unsigned long long)clock() + tId, 0, 0, &curand_state);

  //printf("Generating population delta. v_freeze=%f, kT=%f, mu=%f\n", *v_freeze, *kT, mu);
  //printf("rand_num=[");
  //for (int i=tId; i<n_dbs; i+=stride) {
  for (int i=0; i<n_dbs; i++) {
    printf("genPopulationDelta for tId=%d\n", i);
    // TODO consider replacing expf with __expf for faster perf
    float prob = 1. / ( 1. + expf( ((2.*n[i]-1.)*(v_local[i]+mu) + *v_freeze ) / *kT ));
    float rand_num = curand_uniform(&curand_state);
    //if (randnum[i] < prob) {
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

__device__ void systemEnergy(int n_dbs, float *n, float *v_ext, float *v_ij, float *output)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tId != 0) return;

  // TODO might be able to merge this function with population change energy delta with the similarity
  float *coulomb_v = (float*)malloc(sizeof(float));
  totalCoulombPotential(n_dbs, n, v_ij, coulomb_v);
  __syncthreads();
  cublasSdot(cb_hdl, n_dbs, v_ext, 1, n, 1, output);
  __syncthreads();
  *output *= -1;
  *output += *coulomb_v;
  free(coulomb_v);
}

// Energy change from population change.
__device__ void populationChangeEnergyDelta(int n_dbs, float *dn, float *v_ij, float *v_local, float *output)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tId != 0) return;

  // delta E = -1 * dot(v_local, dn) + dn^T * V_ij * dn
  float *coulomb_v = (float*)malloc(sizeof(float));
  totalCoulombPotential(n_dbs, dn, v_ij, coulomb_v);
  __syncthreads();
  cublasSdot(cb_hdl, n_dbs, v_local, 1, dn, 1, output);
  __syncthreads();
  *output *= -1;
  *output += *coulomb_v;
  free(coulomb_v);
}

// Total potential from Coulombic repulsion in the system.
__device__ void totalCoulombPotential(int n_dbs, float *n, float *v_ij, float *v)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tId != 0) return;

  float alpha=0.5;
  float beta=0;
  float *v_temp = (float*)malloc(n_dbs);
  cublasStatus_t status = cublasSgemv(cb_hdl, CUBLAS_OP_N, n_dbs, n_dbs, &alpha, v_ij, n_dbs, n, 1, &beta, v_temp, 1);
  __syncthreads();
  status = cublasSdot(cb_hdl, n_dbs, n, 1, v_temp, 1, v);
  __syncthreads();
  free(v_temp);
}

__device__ void acceptHop(float *v_diff, float *kT, bool *accept)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tId != 0) return;

  curandState curand_state;
  curand_init((unsigned long long)clock() + tId, 0, 0, &curand_state);

  if (*v_diff <= 0.) {
    *accept = true;
  } else {
    float prob = expf( -(*v_diff) / (*kT) );
    *accept = curand_uniform(&curand_state) < prob;
  }
}

__device__ void hopEnergyDelta(int i, int j, int n_dbs, float *v_local, float *v_ij, float *v_del)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tId == 0)
    *v_del = v_local[i] - v_local[j] - v_ij[IDX2C(i,j,n_dbs)];
}

__device__ void randInt(int cap, int *output)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tId !=0) return;

  curandState curand_state;
  curand_init((unsigned long long)clock() + tId, 0, 0, &curand_state);
  *output = cap;
  while (*output == cap)
    *output = static_cast<int>(cap * curand_uniform(&curand_state));  // floor by cast
}

__device__ void timeStep(int *t, float *kT, float *v_freeze)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tId != 0) return;

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
  // initialize variables & perform pre-calculations
  float kT = 300*constants::Kb; // kT = Boltzmann constant (eV/K) * 300K
  
  float *d_v_ext, *d_v_ij;
  cudaMallocManaged(&d_v_ext, n_dbs*sizeof(float));
  cudaMallocManaged(&d_v_ij, n_dbs*n_dbs*sizeof(float));

  for (int i=0; i<n_dbs; i++) {
    d_v_ext[i] = v_ext[i];
    for (int j=0; j<n_dbs; j++) {
      d_v_ij[IDX2C(i,j,n_dbs)] = v_ij(i,j);
    }
  }

  std::cout << "initializing cublas handle" << std::endl;
  ::initCublasHandle<<<1,1>>>();
  cudaDeviceSynchronize();

  std::cout << "initializing CUDA SimAnneal constants" << std::endl;
  ::initSimAnnealConsts<<<1,1>>>(mu, kT0, kT_step, v_freeze_step);
  cudaDeviceSynchronize();

  std::cout << "invoking CUDA SimAnneal..." << std::endl;
  ::simAnnealAlg<<<1,1>>>(n_dbs, d_v_ext, d_v_ij, t_max, kT);
  cudaDeviceSynchronize();

  std::cout << "destroying cublas handle" << std::endl;
  ::destroyCublasHandle<<<1,1>>>();
  cudaDeviceSynchronize();

  // TODO move results to a form understood by SiQADConn
  
  // clean up
  cudaFree(d_v_ext);
  cudaFree(d_v_ij);
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
