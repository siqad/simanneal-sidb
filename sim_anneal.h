// @file:     sim_anneal.h
// @author:   Samuel
// @created:  2017.08.23
// @editted:  2018-06-13 - Robert
// @license:  Apache License 2.0
//
// @desc:     Simulated annealing physics engine

#ifndef _PHYS_SIM_ANNEAL_H_
#define _PHYS_SIM_ANNEAL_H_

// #include "phys_engine.h"
#include "siqadconn.h"
#include <vector>
#include <deque>
#include <tuple>
#include <memory>
#include <cmath>

#include <boost/thread.hpp>
#include <boost/random.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace constants{
  const float Q0 = 1.602E-19;
  const float PI = 3.14159;
  const float EPS0 = 8.854E-12;
  //const float EPS_SURFACE = 6.35;
  const float Kb = 8.617E-5;
  const float ERFDB = 5E-10;
}

namespace phys {
  namespace ublas = boost::numeric::ublas;

  typedef boost::circular_buffer<ublas::vector<int>> ThreadChargeResults;
  typedef boost::circular_buffer<float> ThreadEnergyResults;
  typedef std::vector<ThreadChargeResults> AllChargeResults;
  typedef std::vector<ThreadEnergyResults> AllEnergyResults;

  // Forward declaration
  class SimAnnealThread;

  struct SimParams
  {
    // runtime params
    int num_threads;            // Number of threads to spawn
    int result_queue_size;      // Number of results to store (per thread)

    // annealing params
    int anneal_cycles;          // Total number of annealing cycles
    float min_T;                // Minimum annealing temperature
    float Kc;                   // 1 / (4 pi eps)
    float kT0;                  // Initial annealing temperature
    float kT_step;              // kT decrement per cycle
    float v_freeze_step;        // Freeze-out voltage increment per cycle

    // physics params
    float mu;                   // Global Fermi level (eV)
    float epsilon_r;            // Relative premittivity on the surface
    float debye_length;         // Debye Length (nm)
    int n_dbs;                  // Number of DBs in the simulation
    std::vector<std::pair<float,float>> db_locs;  // Location of DBs
    ublas::matrix<float> db_r;  // Matrix of distances between all DBs
    ublas::matrix<float> v_ij;  // Matrix of coulombic repulsion between occupied DBs
    ublas::vector<float> v_ext; // External potential influences
  };


  //! Controller class which spawns SimAnneal threads for actual computation.
  class SimAnneal
  {
  public:
    //! Constructor taking the simulation parameters.
    SimAnneal();

    //! Initialize simulation (precomputation, setup common write-out variables,
    //! etc.
    void initialize();

    //! Invoke the desired number of annealers (threads) with the sim_params
    //! stored in the class.
    void invokeSimAnneal();

    //! Calculate system energy of a given configuration, must be exposed 
    //! publically such that the interface can recalculate system energy 
    //! for configurations storage.
    //! TODO Make system energy recalculation optional.
    static float systemEnergy(std::string n_in, int n_dbs);

    // ACCESSORS

    //! Write simulation results of the given SimAnnealThread.
    static void storeResults(SimAnnealThread *annealer, int thread_id);

    //! Return the electron charge location results.
    AllChargeResults& chargeResults() {return charge_results;}

    //! Return the energy results.
    AllEnergyResults& energyResults() {return energy_results;}

    //! Publically accessible simulation parameters
    static SimParams sim_params;

  private:

    //! Calculate the Euclidean distance between points (x1,y1) and (x2,y2).
    float distance(const float &x1, const float &y1, const float &x2, const float &y2);

    //! Calculate the potential between two given point charges.
    float interElecPotential(const float &r);

    // Thread mutex for result storage
    static boost::mutex result_store_mutex;

    // Runtime variables
    std::vector<boost::thread> anneal_threads;  //! threads spawned

    // Simulation variables
    static float db_distance_scale;     //! convert db distances to m TODO make this configurable in user settings

    // Write-out variables
    static AllChargeResults charge_results; //Vector for storing db_charges
    static AllEnergyResults energy_results; //Vector for storing config_energies
    //static std::vector<int> numElecStore;
  };

  class SimAnnealThread
  {
  public:

    // constructor
    SimAnnealThread(const int t_thread_id);

    // destructor
    ~SimAnnealThread() {};

    // run simulation
    void run();

    int n_elec=0;               // number of doubly occupied DBs
    ublas::vector<int> n;       // electron configuration at the current time-step
    std::vector<int> occ;       // indices of dbs, first n_elec indices are occupied
    ThreadChargeResults db_charges;       // charge configuration history
    ThreadEnergyResults config_energies;  // energy history corresponding to db_charges

    int thread_id;              // the thread id of each class object

    // VARIABLES
    //const float har_to_ev = 27.2114; // hartree to eV conversion factor

  private:

    // Generate the delta in population.
    ublas::vector<int> genPopDelta(bool &changed);

    // simmulated annealing accessor
    void anneal();

    // perform an electron hop from one DB to another
    void performHop(const int &from_ind, const int &to_ind);

    // advance time-step
    void timeStep();

    // CALCULATIONS
    float systemEnergy() const;
    float totalCoulombPotential(ublas::vector<int> &config) const;
    float hopEnergyDelta(const int &i, const int &j);

    // ACCEPTANCE FUNCTIONS
    bool acceptHop(const float &v_diff); // acceptance function for hopping
    bool evalProb(const float &prob); // generate true or false based on given probaility

    int randInt(const int &min, const int &max);

    // boost random number generator
    boost::random::uniform_real_distribution<float> dis01;
    boost::random::mt19937 rng;

    // other variables used for calculations
    int t=0;                      // current annealing cycle
    float kT, v_freeze;           // current annealing temperature, freeze out potential
    ublas::vector<float> v_local; // local potetial at each site

    float E_sys;                  // energy of the system
  };
}


#endif
