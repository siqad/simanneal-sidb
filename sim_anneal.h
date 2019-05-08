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
  // lattice
  const FPType lat_a = 3.84;  // lattice vector in x, angstroms (intra dimer row)
  const FPType lat_b = 7.68;  // lattice vector in y, angstroms (inter dimer row)
  const FPType lat_c = 2.25;  // dimer pair separation, angstroms

  // physics
  const FPType Q0 = 1.602E-19;
  const FPType PI = 3.14159;
  const FPType EPS0 = 8.854E-12;
  const FPType Kb = 8.617E-5;
  const FPType ERFDB = 5E-10;

  // simulation

  // Allowed headroom in eV for physically invalid configurations to still be 
  // considered "probably valid", the validity will be re-determined during export.
  // Typical error is 1E-4 or lower, so this should be plenty enough headroom.
  const FPType POP_STABILITY_ERR = 1E-3;
  const FPType RECALC_STABILITY_ERR = 1E-6;
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
    FPType alpha;                // T(t) = alpha * T(t-1) + T_min
    FPType T_init;               // Initial annealing temperature
    FPType T_min;                // Minimum annealing temperature

    // v_freeze params
    // v_freeze increases from the v_freeze_init value to the v_freeze_threshold
    // value over a course of v_freeze_cycles. After v_freeze_cycles is reached,
    // the physical validity of the electron configuration is checked for 
    // phys_validity_check_cycles consecutive cycles. If a majority of those 
    // cycles have physically valid layouts, then v_freeze stays the same for 
    // another v_freeze_cycles until the check is performed again. Otherwise, 
    // v_freeze is reset to v_freeze_init.
    FPType v_freeze_init;        // Initial freeze-out voltage
    FPType v_freeze_threshold;   // Final freeze-out voltage
    FPType v_freeze_reset;       // Freeze-out voltage to reset to
    int v_freeze_cycles;        // Cycles per v_freeze_period, set to -1 to set to the same as anneal_cycles
    int phys_validity_check_cycles;
    bool strategic_v_freeze_reset;
    bool reset_T_during_v_freeze_reset;

    // calculated params (from annealing params)
    FPType Kc;                   // 1 / (4 pi eps)
    FPType kT_min;               // Kb * T_min
    FPType v_freeze_step;        // (v_freeze_threshold - v_freeze_init) / v_freeze_cycles

    // physics params
    FPType mu;                   // Global Fermi level (eV)
    FPType epsilon_r;            // Relative premittivity on the surface
    FPType debye_length;         // Debye Length (nm)
    int n_dbs;                  // Number of DBs in the simulation
    std::vector<std::pair<FPType,FPType>> db_locs;  // Location of DBs
    ublas::matrix<FPType> db_r;  // Matrix of distances between all DBs
    ublas::matrix<FPType> v_ij;  // Matrix of coulombic repulsion between occupied DBs
    ublas::vector<FPType> v_ext; // External potential influences
  };

  // ElecChargeConfigs that are written to the shared simulation results list
  // which is later processed for export.
  struct ElecChargeConfigResult
  {
    ElecChargeConfigResult() {};
    ElecChargeConfigResult(ublas::vector<int> config, bool population_possibly_stable, FPType system_energy)
      : config(config), population_possibly_stable(population_possibly_stable), system_energy(system_energy) {};
    
    bool isResult() {return config.size() > 0;}

    ublas::vector<int> config;
    bool population_possibly_stable=false;
    FPType system_energy;
  };

  struct TimeInfo
  {
    double wall_time=-1;
    double cpu_time=-1;
  };

  // typedefs
  typedef boost::circular_buffer<ElecChargeConfigResult> ThreadChargeResults;
  typedef boost::circular_buffer<FPType> ThreadEnergyResults;
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
    static FPType systemEnergy(const std::string &n_in, int n_dbs);
    static FPType systemEnergy(const ublas::vector<int> &n_in, bool qubo=false);

    //! Return whether the given configuration population is valid. Population
    //! validity is evaluated based on the following criteria:
    //! 1. occupied DBs have local energy lower than mu;
    //! 2. unoccupied DBs have local energy higher than mu.
    static bool populationValidity(const ublas::vector<int> &n_in);

    //! Return whether the given configuration is locally minimal. In other 
    //! words, whether there are lower energy states that can be accessed from a
    //! single hopping event, even if that lower energy state itself is 
    //! physically invalid.
    static bool locallyMinimal(const ublas::vector<int> &n_in);

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
    FPType distance(const int &i, const int &j);

    //! Calculate the potential between two given point charges.
    FPType interElecPotential(const FPType &r);

    //! Return the energy difference for a configuration if an electron hopped
    //! from site i to site j.
    static FPType hopEnergyDelta(ublas::vector<int> n_in, const int &from_ind,
        const int &to_ind);

    // Thread mutex for result storage
    static boost::mutex result_store_mutex;

    // Runtime variables
    std::vector<boost::thread> anneal_threads;  //! threads spawned

    // Simulation variables
    static FPType db_distance_scale;     //! convert db distances to m TODO make this configurable in user settings

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
    void genPopDelta(ublas::vector<int> &dn, bool &changed);

    // simmulated annealing accessor
    void anneal();

    // perform an electron hop from one DB to another
    void performHop(const int &from_ind, const int &to_ind);

    // advance time-step
    void timeStep();

    // CALCULATIONS
    FPType systemEnergy() const;
    FPType totalCoulombPotential(ublas::vector<int> &config) const;
    FPType hopEnergyDelta(const int &i, const int &j);

    // Return whether the current electron population is valid with an error
    // headroom to account for FPTypeing point drifts during energy updatse.
    bool populationValid(const FPType &err_headroom) const;

    // Return whether the current configuration is physically valid.
    bool isPhysicallyValid();

    // ACCEPTANCE FUNCTIONS
    bool acceptHop(const FPType &v_diff); // acceptance function for hopping
    bool evalProb(const FPType &prob); // generate true or false based on given probaility

    int randInt(const int &min, const int &max);

    // boost random number generator
    boost::random::uniform_real_distribution<FPType> dis01;
    boost::random::mt19937 rng;

    // keep track of stats
    int n_elec=0;               // number of doubly occupied DBs
    ublas::vector<int> n;       // electron configuration at the current time-step
    std::vector<int> occ;       // indices of dbs, first n_elec indices are occupied

    // other variables used for calculations
    int t=0;                      // current annealing cycle
    int t_freeze=0;               // current v_freeze cycle
    FPType kT, v_freeze;           // current annealing temperature, freeze out potential
    ublas::vector<FPType> v_local; // local potetial at each site

    int t_phys_validity_check=0;
    PopulationSchedulePhase pop_schedule_phase;
    int phys_valid_count;
    int phys_invalid_count;

    // keep track of the suggested config - physically valid ground state if 
    // possible, invalid ground state otherwise. Only keeps track if 
    ublas::vector<int> n_valid_gs;
    ublas::vector<int> n_invalid_gs;
    FPType E_sys_valid_gs = 0;
    FPType E_sys_invalid_gs = 0;

    FPType E_sys;                  // energy of the system
  };
}


#endif
