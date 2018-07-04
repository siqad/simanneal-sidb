// @file:     sim_anneal.h
// @author:   Samuel
// @created:  2017.08.23
// @editted:  2018-06-13 - Robert
// @license:  GNU LGPL v3
//
// @desc:     Simulated annealing physics engine

// #include "phys_engine.h"
#include "siqadconn.h"
#include <vector>
#include <deque>
#include <tuple>
#include <memory>
#include <cmath>
#include <thread>
#include <mutex>

#include <boost/random.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>

#ifndef __SIM_ANNEAL_H__
#define __SIM_ANNEAL_H__


#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif


// 0-based index for cuBLAS
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

namespace constants{
  const float Q0 = 1.602E-19;
  const float PI = 3.14159;
  const float EPS0 = 8.854E-12;
  const float EPS_SURFACE = 6.35;
  const float Kb = 8.617E-5;
  const float ERFDB = 5E-10;
  const float DB_SCALE = 1E-10;
}

// CUDA functions

__global__ void simAnnealParallel(int stream_id, int results_to_return, float *n_out);

// Initialize cublas handles in parallel
__global__ void initCublasHandles(int num_streams);

// Destroy cublas handles in parallel
__global__ void destroyCublasHandles(int num_streams);

// Initialize simanneal constants
__global__ void initDeviceVars(int num_streams, float n_dbs_in, float debye_length, float mu_in, 
    float kT0_in, float kT_step_in, float v_freeze_step_in, int t_max_in, 
    float *v_ext_in, float *db_locs);

// Initialize v_ij array
__global__ void initVij(int n_dbs, float debye_length, float *db_locs, float *v_ij);

// Destroy device variables
__global__ void cleanUpDeviceVars(int num_streams);

// Initialize v_local
__device__ void initVLocal(cublasHandle_t cublas_handle, int n_dbs, float *n, float *v_ext, float *v_local);

// Update v_local after hopping from site i to site j.
__global__ void updateVLocal(int from_ind, int to_ind, int n_dbs, float *v_local);

// Generate population delta (array of -1, 0 or 1 indicating the change in electron count at each site).
__global__ void genPopulationDelta(int n_dbs, float *n, float *v_local, float *v_freeze, float *kT, float *dn, bool *pop_changed);
__device__ void genPopulationDelta(curandState *curand_state, int n_dbs, float *n, float *v_local, float *v_freeze, float *kT, float *dn, bool *pop_changed);

// Total system energy including Coulombic repulsion and external voltage.
__device__ void systemEnergy(cublasHandle_t cublas_handle, int n_dbs, float *n, float *v_ext, float *output);

// Energy change from population change.
__device__ void populationChangeEnergyDelta(cublasHandle_t cublas_handle, int n_dbs, float *dn, float *v_local, float *output);

// Total potential from Coulombic repulsion in the system.
__device__ void totalCoulombPotential(cublasHandle_t cublas_handle, int n_dbs, float *n, float *v);

// Hop acceptance function.
__device__ void acceptHop(curandState *curand_state, float *v_diff, float *kT, bool *accept);

// Energy change due to an electron hopping from site i to j.
__device__ void hopEnergyDelta(int i, int j, int n_dbs, float *v_local, float *v_del);

// Generate an array of random floats within the range 0.0 < num <= 1.0.
__device__ void randomFloats(int len, float *arr);

// Generate an array of random ints within the range 0 <= num < cap.
__device__ void randInt(curandState *state, int cap, int *output);

// Time step.
__device__ void timeStep(int *t, float *kT, float *v_freeze);

// Return configuration to the array provided by host.
__device__ void returnN(float *n_local, float *n_out);



namespace phys {

  namespace ublas = boost::numeric::ublas;
  class SimAnneal
  {
  public:

    // constructor
    SimAnneal(const int thread_id);

    // destructor
    ~SimAnneal() {};

    // run simulation
    void runSim();

    // run CUDA version of simanneal
    void runSimCUDA();

    //static std::vector< boost::circular_buffer<ublas::vector<int>> > chargeStore; //Vector for storing db_charges
    //static std::vector< boost::circular_buffer<float> > energyStore; //Vector for storing config_energies
    static std::vector< std::vector<ublas::vector<int>> > chargeStore; //Vector for storing db_charges
    static std::vector< std::vector<float> > energyStore; //Vector for storing config_energies
    static std::vector<int> numElecStore;

    //Total calculations
    float distance(const float &x1, const float &y1, const float &x2, const float &y2);
    float interElecPotential(const float &r);

    // keeping track of electron configurations and other house keeping vars
    static int n_dbs; // number of dbs
    int n_elec=0; // number of doubly occupied DBs
    static ublas::matrix<float> db_r;  // distance between all dbs
    ublas::vector<int> n;       // electron configuration at the current time-step
    std::vector<int> occ;       // indices of dbs, first n_elec indices are occupied
    static ublas::vector<float> v_ext; // keep track of voltages at each DB
    static ublas::matrix<float> v_ij;     // coulombic repulsion
    static int result_queue_size;
    static std::vector<std::pair<float,float>> db_locs; // location of free dbs
    //boost::circular_buffer<ublas::vector<int>> db_charges;
    std::vector<ublas::vector<int>> db_charges;
    //boost::circular_buffer<float> config_energies;  // energies in line with db_charges
    std::vector<float> config_energies;  // energies in line with db_charges

    static int num_threads;    // number of threads used in simmulation
    int threadId;              // the thread id of each class object

    // other variables used for calculations
    static float kT0, kT_step, v_freeze_step;

    // handy constants or variables from problem file
    static float Kc;           // 1 / (4 pi eps)
    static float debye_length; // Silicon intrinsic Debye length in m
    static float mu;          // global bias

    //Other variables used for CALCULATIONS
    static int t_max;          // keep track of annealing cycles

    // VARIABLES
    //const float har_to_ev = 27.2114; // hartree to eV conversion factor
    const float db_distance_scale = 1E-10; // TODO move this to xml

  private:

    // determine change in population
    ublas::vector<int> genPopDelta();

    // simmulated annealing accessor
    void simAnneal();

    // perform an electron hop from one DB to another
    void performHop(int from_ind, int to_ind);

    // advance time-step
    void timeStep();

    // CALCULATIONS
    float systemEnergy();
    static float systemEnergy(ublas::vector<int> config);
    float totalCoulombPotential(ublas::vector<int> config);
    float hopEnergyDelta(const int &i, const int &j);

    // ACCEPTANCE FUNCTIONS
    bool acceptPop(int db_ind);
    bool acceptHop(float v_diff); // acceptance function for hopping
    bool evalProb(float prob); // generate true or false based on given probaility

    // OTHER ACCESSORS
    int getRandOccInd(int charge);

    // print the current charge configuration into cout
    void printCharges();

    // boost random number generator
    boost::random::uniform_real_distribution<float> dis01;
    boost::random::mt19937 rng;

    // other variables used for calculations
                                // temperature, time
    int t=0;                    // keep track of annealing cycles
    float kT, v_freeze;         // freeze out potential (pushes
                                // out population transition probability)
    ublas::vector<float> v_local;

    float E_sys;                  // energy of the system

/*
    //Vars used in restarting
    float E_best;                 // best energy of the system thus far
    boost::numeric::ublas::vector<int> n_best; // electron configuration at the lowest recorded energy
    int steadyPopCount;
    bool firstBest;
*/
  };
}

#endif
