// @file:     sim_anneal.h
// @author:   Samuel
// @created:  2017.08.23
// @editted:  2019.02.01
// @license:  Apache License 2.0
//
// @desc:     Simulated annealing physics engine

#ifndef _PHYS_SIM_ANNEAL_H_
#define _PHYS_SIM_ANNEAL_H_

// #include "phys_engine.h"
#include "global.h"
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

  // enums
  enum TemperatureSchedule{LinearSchedule, ExponentialSchedule};

  // Forward declaration
  class SimAnnealThread;

  struct SimParams
  {
    // runtime params
    int num_threads;            // Number of threads to spawn
    int result_queue_size;      // Number of results to store (per thread)

    // annealing params
    int anneal_cycles;          // Total number of annealing cycles
    int preanneal_cycles;       // Initial cycles where temperature doesn't change
    TemperatureSchedule T_schedule;
    float alpha;                // T(t) = alpha * T(t-1) + T_min
    float T_init;               // Initial annealing temperature
    float T_min;                // Minimum annealing temperature

    // v_freeze params
    // v_freeze increases from the v_freeze_init value to the v_freeze_threshold
    // value over a course of v_freeze_cycles. After v_freeze_cycles is reached,
    // the physical validity of the electron configuration is checked for 
    // phys_validity_check_cycles consecutive cycles. If a majority of those 
    // cycles have physically valid layouts, then v_freeze stays the same for 
    // another v_freeze_cycles until the check is performed again. Otherwise, 
    // v_freeze is reset to v_freeze_init.
    float v_freeze_init;        // Initial freeze-out voltage
    float v_freeze_threshold;   // Final freeze-out voltage
    float v_freeze_reset;       // Freeze-out voltage to reset to
    int v_freeze_cycles;        // Cycles per v_freeze_period, set to -1 to set to the same as anneal_cycles
    int phys_validity_check_cycles;
    bool strategic_v_freeze_reset;
    bool reset_T_during_v_freeze_reset;

    // calculated params (from annealing params)
    float Kc;                   // 1 / (4 pi eps)
    float kT_min;               // Kb * T_min
    float v_freeze_step;        // (v_freeze_threshold - v_freeze_init) / v_freeze_cycles

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

  struct TimeInfo
  {
    double wall_time=-1;
    double cpu_time=-1;
  };

  // typedefs
  typedef boost::circular_buffer<ublas::vector<int>> ThreadChargeResults;
  typedef boost::circular_buffer<float> ThreadEnergyResults;
  typedef std::vector<ThreadChargeResults> AllChargeResults;
  typedef std::vector<ThreadEnergyResults> AllEnergyResults;
  typedef std::vector<double> AllCPUTimes;
  typedef std::vector<ublas::vector<int>> AllSuggestedConfigResults;

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
    static float systemEnergy(const std::string &n_in, int n_dbs);

    //! Return whether the given configuration is physically valid. Validity is 
    //! evaluated based on the following criteria:
    //! 1. occupied DBs have local energy lower than mu;
    //! 2. unoccupied DBs have local energy higher than mu.
    static bool isPhysicallyValid(const std::string &n_in, int n_dbs);

    // ACCESSORS

    //! Write simulation results of the given SimAnnealThread.
    static void storeResults(SimAnnealThread *annealer, int thread_id);

    //! Return the electron charge location results.
    AllChargeResults &chargeResults() {return charge_results;}

    //! Return the energy results.
    AllEnergyResults &energyResults() {return energy_results;}

    //! Return CPU timing results.
    AllCPUTimes &CPUTimeingResults() {return cpu_times;}

    //! Return suggested config results.
    AllSuggestedConfigResults &suggestedConfigResults() {return suggested_config_results;}

    //! Publically accessible simulation parameters
    static SimParams sim_params;

  private:

    //! Calculate the Euclidean distance between the i th and j th DBs in the 
    //! db_locs array.
    float distance(const int &i, const int &j);

    //! Calculate the potential between two given point charges.
    float interElecPotential(const float &r);

    // Thread mutex for result storage
    static boost::mutex result_store_mutex;

    // Runtime variables
    std::vector<boost::thread> anneal_threads;  //! threads spawned

    // Simulation variables
    static float db_distance_scale;     //! convert db distances to m TODO make this configurable in user settings

    // Write-out variables
    static AllChargeResults charge_results; // vector for storing db_charges
    static AllEnergyResults energy_results; // vector for storing config_energies
    static AllCPUTimes cpu_times;           // vector for storing time information
    static AllSuggestedConfigResults suggested_config_results;
  };

  class SimAnnealThread
  {
  public:

    enum PopulationSchedulePhase{PopulationUpdateMode, PhysicalValidityCheckMode,
      PopulationUpdateFinished};

    // constructor
    SimAnnealThread(const int t_thread_id);

    // destructor
    ~SimAnnealThread() {};

    // run simulation
    void run();

    // return the total CPU time in seconds
    double CPUTime()
    {
      // CPU time
      struct timespec curr_cpu_time;
      clockid_t thread_clock_id;
      pthread_getcpuclockid(pthread_self(), &thread_clock_id);
      clock_gettime(thread_clock_id, &curr_cpu_time);
      return (double) curr_cpu_time.tv_sec + 1e-9 * curr_cpu_time.tv_nsec;
    }

    // return the physically valid ground state of this thread.
    ublas::vector<int> suggestedConfig() {
      return n_valid_gs.empty() ? n_invalid_gs : n_valid_gs;
    }

    int thread_id;              // the thread id of each class object
    ThreadChargeResults db_charges;       // charge configuration history
    ThreadEnergyResults config_energies;  // energy history corresponding to db_charges

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

    // Return whether the current configuration is physically valid.
    bool isPhysicallyValid();

    // ACCEPTANCE FUNCTIONS
    bool acceptHop(const float &v_diff); // acceptance function for hopping
    bool evalProb(const float &prob); // generate true or false based on given probaility

    int randInt(const int &min, const int &max);

    // boost random number generator
    boost::random::uniform_real_distribution<float> dis01;
    boost::random::mt19937 rng;

    // keep track of stats
    int n_elec=0;               // number of doubly occupied DBs
    ublas::vector<int> n;       // electron configuration at the current time-step
    std::vector<int> occ;       // indices of dbs, first n_elec indices are occupied

    // other variables used for calculations
    int t=0;                      // current annealing cycle
    int t_freeze=0;               // current v_freeze cycle
    float kT, v_freeze;           // current annealing temperature, freeze out potential
    ublas::vector<float> v_local; // local potetial at each site

    int t_phys_validity_check=0;
    PopulationSchedulePhase pop_schedule_phase;
    int phys_valid_count;
    int phys_invalid_count;

    // keep track of the suggested config - physically valid ground state if 
    // possible, invalid ground state otherwise. Only keeps track if 
    ublas::vector<int> n_valid_gs;
    ublas::vector<int> n_invalid_gs;
    float E_sys_valid_gs = 0;
    float E_sys_invalid_gs = 0;

    float E_sys;                  // energy of the system
  };
}


#endif
