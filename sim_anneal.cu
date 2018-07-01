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

#include <cuda_runtime.h>
#include "cublas_v2.h"

//#define STEADY_THREASHOLD 700       //arbitrary value used in restarting

using namespace phys;

std::mutex siqadMutex;

// Total system energy including Coulombic repulsion and external voltage.
// TODO get this working on multiple threads. Not supposed to add to the same variable *v as this isn't thread-safe.
__global__ void systemEnergy(float *v, int n_dbs, float *n, float *v_ext, float *v_ij)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i=index; i<n_dbs; i+=stride) {
    *v -= v_ext[i] * n[i];
    for (int j=i+1; j<n_dbs; j++)
      *v += v_ij[i*n_dbs + j] * n[i] * n[j];
  }
}

// Total potential from Coulombic repulsion in the system.
// TODO get this working on multiple threads. Not supposed to add to the same variable *v as this isn't thread-safe.
__global__ void totalCoulombPotential(float *v, int n_dbs, float *dn, float *v_ij)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  float temp;
  for (int i=index; i<n_dbs; i+=stride) {
    temp = 0;
    for (int j=0; j<n_dbs; j++)
      temp += v_ij[i*n_dbs+j] * dn[j];
    *v += 0.5 * temp * dn[i];
  }
}

// a matrix vector product where the matrix must be square with dimension nxn.
/*__device__ void matrixVectorProd(int n, float *prod, float *A, float *a)
{
  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      // TODO this initialization is probably wrong, figure out proper way
      if (i==0 && j==0)
        *prod[i] = 0;
      *prod[i] += A[i*n+j] * a[j];
    }
  }
}

// a dot product between two vectors v and u.
__device__ void dotProd(int n, float *prod, float *a, float *b)
{
  for (int i=0; i<n; i++) {
    // TODO this initialization is probably wrong, figure out proper way
    if (i==0)
      *prod = 0;
    *prod += a[i] * b[i];
  }
}*/

__global__ void hopEnergyDelta(float *v_del, int n_dbs, float *v_local, float *v_ij, int i, int j)
{
  *v_del = v_local[i] - v_local[j] - v_ij[i*n_dbs+j];
}


//Global method for writing to vectors (global in order to avoid thread clashing).
void writeStore(SimAnneal *object, int threadId){
  siqadMutex.lock();

  object->chargeStore[threadId] = object->db_charges;
  object->energyStore[threadId] = object->config_energies;
  object->numElecStore[threadId] = object->n_elec;

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

  /* NOTE cuBLAS attempt, incomplete
  cudaError_t cuda_stat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  float *dev_n, *dev_v_local, *dev_v_ij, *dev_v_temp;

  cuda_stat = cudaMalloc((void**)&dev_n, n_dbs*sizeof(float));
  if (cuda_stat != cudaSuccess) {
    std::cout << "device memory allocation for n failed" << std::endl;
    return;
  }
  cuda_stat = cudaMalloc((void**)&dev_v_ij, n_dbs*n_dbs*sizeof(float));
  if (cuda_stat != cudaSuccess) {
    std::cout << "device memory allocation for v_ij failed" << std::endl;
    return;
  }

  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLAS initialization failed" << std::endl;
    return;
  }

  stat = cublasSetMatrix(n_dbs, n_dbs, sizeof(*n_arr), n_arr, dev_n, n_dbs);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cout << "data download failed" << std::endl;
    cudaFree(dev_n);
    cublasDestroy(handle);
    return;
  }
  // TODO set vector

  // TODO matrix multiplication
  float alf=1.0;
  float beta=1.0;
  stat = cublasSgemv(handle, CUBLAS_OP_T, n_dbs, n_dbs, &alf, dev_v_ij, n_dbs, dev_n, 1, &beta, v_temp, 1);

  stat = cublasGetMatrix(n_dbs, n_dbs, sizeof(*n_arr), dev_n, n_dbs, n_arr, n_dbs);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cout << "data upload failed" << std::endl;
    cudaFree(dev_n);
    cublasDestroy(handle);
    return;
  }

  cudaFree(dev_n);
  cublasDestroy(handle);
  */
  
  // arrays for CUDA code
  float *n_arr, *v_ext_arr, *v_ij_arr, *cuda_v, *dn_arr;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&n_arr, n_dbs*sizeof(float));
  cudaMallocManaged(&v_ext_arr, n_dbs*sizeof(float));
  cudaMallocManaged(&v_ij_arr, n_dbs*n_dbs*sizeof(float));
  cudaMallocManaged(&cuda_v, sizeof(float));
  cudaMallocManaged(&dn_arr, n_dbs*sizeof(float));

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



      // NOTE CUDA test
      float c_ver = totalCoulombPotential(dn);

      // copy vector data to array
      for (int i=0; i<n_dbs; i++) {
        dn_arr[i] = dn[i];
        n_arr[i] = n[i];
        v_ext_arr[i] = v_ext[i];
        for (int j=0; j<n_dbs; j++) {
          v_ij_arr[i*n_dbs + j] = v_ij(i,j);
        }
      }
      *cuda_v = 0;
      ::totalCoulombPotential<<<1, 1>>>(cuda_v, n_dbs, dn_arr, v_ij_arr);

      cudaDeviceSynchronize();

      std::cout << "total coulomb potential c++ : " << c_ver << std::endl;
      std::cout << "total coulomb potential cuda: " << *cuda_v << std::endl;

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

  // Run kernel on 1M elements on the GPU
  *cuda_v = 0;
  ::systemEnergy<<<1, 1>>>(cuda_v, n_dbs, n_arr, v_ext_arr, v_ij_arr);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  std::cout << "Host systemEnergy()=" << systemEnergy() << std::endl;
  std::cout << "CUDA systemEnergy()=" << *cuda_v << std::endl;

  // Free memory
  cudaFree(n_arr);
  cudaFree(v_ext_arr);
  cudaFree(v_ij_arr);
  cudaFree(cuda_v);
  cudaFree(dn_arr);

  writeStore(this, threadId);
}










ublas::vector<int> SimAnneal::genPopDelta()
{
  ublas::vector<int> dn(n_dbs);
  for (unsigned i=0; i<n.size(); i++) {
    //float prob = 1. / ( 1 + exp( ((2*n[i]-1)*v_local[i] + v_freeze) / kT ) );
    float prob = 1. / ( 1 + exp( ((2*n[i]-1)*(v_local[i] + mu) + v_freeze) / kT ) );
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
