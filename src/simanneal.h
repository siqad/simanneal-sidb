// @file:     sim_anneal.h
// @author:   Samuel
// @created:  2017.08.23
// @editted:  2019.02.01
// @license:  Apache License 2.0
//
// @desc:     Simulated annealing physics engine

#ifndef _PHYS_SIMANNEAL_H_
#define _PHYS_SIMANNEAL_H_

#include "global.h"
#include "hop_selector.h"
#include "libs/siqadconn/src/siqadconn.h"
#include "physical_repair.h"
#include "refinement.h"
#include "search_rng.h"
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

//#include <boost/thread.hpp>
#include <boost/random.hpp>
#include <boost/nondet_random.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace constants{
  // energy band diagram
  const FPType eta = 0.59;    // TODO enter true value; energy difference between (0/-) and (+/0) levels

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
  enum class PopulationBackend {
    Auto,
    Portable,
    Accelerate,
    OpenBLAS,
    OpenBLASSymmetric
  };
  enum class SearchProfile { Legacy, Optimized };
  enum class RandomBackend { Auto, MT, PCG32 };
  enum class FeatureSetting { ProfileDefault, Disabled, Enabled };

  struct SearchStats {
    std::uint64_t executed_restarts = 0, repair_attempts = 0,
                  repair_budget_exhaustions = 0;
    bool singleton_used = false;
    int refinement_center_offset = 0;
    refinement::Stats refinement;
  };

  // Forward declaration
  class SimAnnealThread;

  typedef std::pair<FPType,FPType> EuclCoord;
  typedef std::vector<int> LatCoord;
  //typedef std::vector<EuclCoord2D> DBLocs;
  struct EuclCoord3d {
    FPType x;
    FPType y;
    FPType z;
    EuclCoord3d(FPType t_x, FPType t_y, FPType t_z): x(t_x), y(t_y), z(t_z) {};
  };

  struct SimParams
  {
    SimParams() {};

    void setDBLocs(const std::vector<EuclCoord> &);

    void setDBLocs(const std::vector<LatCoord> &, const phys::LatticeVector &);

    static EuclCoord latToEuclCoord(const int &n, const int &m, const int &l, const phys::LatticeVector &lat_unit_cell);

    // Clear old fixed charges and set new ones
    void setFixedCharges(const std::vector<EuclCoord3d> &t_fc_locs, 
      const std::vector<FPType> &t_fcs, const std::vector<FPType> &t_fc_eps_rs,
      const std::vector<FPType> &t_fc_lambdas);

    // used for alpha and v_freeze_cycles calculation
    int anneal_cycles=10000;      // Total number of annealing cycles
    FPType T_e_inv_point=0.09995; // Where in schedule does T = 1/e * T_0
    FPType v_freeze_end_point=0.4;// Where in schedule does v_freeze stop growing

    // runtime params
    int num_instances=-1;         // Independent restarts (legacy name)
    int num_workers=0;            // Active workers; 0 selects hardware concurrency
    PopulationBackend population_backend=PopulationBackend::Auto; // Build-selected dense backend
    SearchProfile search_profile = SearchProfile::Legacy;
    RandomBackend random_backend = RandomBackend::Auto;
    FeatureSetting repair = FeatureSetting::ProfileDefault;
    FeatureSetting singleton_shortcut = FeatureSetting::ProfileDefault;
    bool probability_shortcuts = true;
    bool transient_domain_mask = false;
    refinement::Options refinement_options;
    // Resolved by solver initialization; callers must use effectiveParams().
    bool repair_enabled = false, singleton_enabled = false;
    std::vector<unsigned char> final_domains;
    bool record_history=false;    // Optional bounded per-cycle diagnostic history
    bool deterministic_seed=false;
    std::uint64_t random_seed=0; // Base seed must fit uint32_t for legacy MT32 streams
    static std::uint64_t parseRandomSeed(const std::string &value) {
      if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("random_seed must be random or an integer in [0,4294967295]");
      // Validate during accumulation, including arbitrarily long decimal input.
      std::uint64_t seed = 0;
      const std::uint64_t limit = std::numeric_limits<std::uint32_t>::max();
      for (const char digit : value) {
        const unsigned next = digit - '0';
        if (seed > (limit - next) / 10)
          throw std::invalid_argument("random_seed must be in [0,4294967295]");
        seed = seed * 10 + next;
      }
      return seed;
    }
    float result_queue_factor=.1; // Number of results to store (per thread)
    int result_queue_size;        // Number of results to store (per thread)
    int hop_attempt_factor=5;     // total hop attempt = hop_attempt_factor * (num_occ - num_vac)

    // Experimental target selection. Uniform preserves the original RNG path.
    HopSelection hop_selection=UniformHop;
    int hop_neighbors=16;
    FPType hop_length_nm=2.0;
    FPType hop_radius_nm=4.0;
    FPType hop_global_probability=0.2;
    HopNeighborhood hop_neighborhood;

    // annealing params
    int preanneal_cycles=0;       // Initial cycles where temperature doesn't change
    TemperatureSchedule T_schedule=ExponentialSchedule;
    FPType alpha;                 // T(t) = alpha * T(t-1) + T_min
    FPType T_init=500;            // Initial annealing temperature
    FPType T_min=2;               // Minimum annealing temperature

    // v_freeze params
    // v_freeze increases from the v_freeze_init value to the v_freeze_threshold
    // value over a course of v_freeze_cycles. After v_freeze_cycles is reached,
    // the physical validity of the electron configuration is checked for 
    // phys_validity_check_cycles consecutive cycles. If a majority of those 
    // cycles have physically valid layouts, then v_freeze stays the same for 
    // another v_freeze_cycles until the check is performed again. Otherwise, 
    // v_freeze is reset to v_freeze_init.
    FPType v_freeze_init=-1;     // Initial freeze-out voltage
    FPType v_freeze_threshold=4; // Final freeze-out voltage
    FPType v_freeze_reset=-1;    // Freeze-out voltage to reset to
    int v_freeze_cycles;        // Cycles per v_freeze_period, set to -1 to set to the same as anneal_cycles
    int phys_validity_check_cycles=10;
    bool strategic_v_freeze_reset=false;
    bool reset_T_during_v_freeze_reset=false;

    // calculated params (from annealing params)
    FPType Kc;                   // 1 / (4 pi eps)
    FPType kT_min;               // Kb * T_min
    FPType v_freeze_step;        // (v_freeze_threshold - v_freeze_init) / v_freeze_cycles

    // physics params
    FPType mu=-0.25;             // Global Fermi level (eV)
    FPType eps_r=5.6;        // Relative premittivity on the surface
    FPType debye_length=5.0;     // Debye Length (nm)
    int n_dbs = 0;               // Number of DBs in the simulation
    phys::LatticeVector lat_vec;
    std::vector<std::pair<FPType,FPType>> db_locs;  // Location of DBs
    ublas::matrix<FPType> db_r;  // Matrix of distances between all DBs
    ublas::matrix<FPType> v_ij;  // Matrix of coulombic repulsion between occupied DBs
    bool population_finite_matrix=true; // Recorded while initialize fills the shared matrix
    ublas::vector<FPType> v_ext; // External potential influences

    // fixed charges (idea is to use them for defects)
    // std::vector<EuclCoord3d> fixed_charge_locs;
    // std::vector<FPType> fixed_charges;        // fixed charges
    // std::vector<FPType> fixed_charge_eps_rs;  // relative permittivities of fixed charges
    // std::vector<FPType> fixed_charge_lambdas; // thomas-fermi screening length of fixed charges

    ublas::vector<FPType> v_fc; // External potential influences from fixed charges
  };

  // ChargeConfigs that are written to the shared simulation results list which 
  // is later processed for export.
  struct ChargeConfigResult
  {
    ChargeConfigResult() 
      : initialized(false), config(ublas::vector<int>()), 
        pop_likely_stable(false), system_energy(0) {};
    ChargeConfigResult(const ublas::vector<int> &config, bool pop_likely_stable,
        FPType system_energy)
      : initialized(true), config(config), pop_likely_stable(pop_likely_stable), 
        system_energy(system_energy) {};
    
    bool isResult() const {return config.size() > 0;}

    bool initialized = false;
    bool refinement_result = false;
    bool repair_attempted = false, repair_budget_exhausted = false;
    ublas::vector<int> config;
    bool pop_likely_stable;
    FPType system_energy;
  };

  struct TimeInfo
  {
    double wall_time=-1;
    double cpu_time=-1;
  };

  // typedefs
  typedef boost::circular_buffer<ChargeConfigResult> ThreadChargeResults;
  typedef boost::circular_buffer<FPType> ThreadEnergyResults;
  typedef std::vector<ThreadChargeResults> AllChargeResults;
  typedef std::vector<ThreadEnergyResults> AllEnergyResults;
  typedef std::vector<ChargeConfigResult> SuggestedResults;
  //typedef std::vector<double> AllCPUTimes;

  //! Controller class which spawns SimAnneal threads for actual computation.
  class SimAnneal
  {
  public:

    //! Constructor taking the simulation parameters.
    SimAnneal(SimParams &sparams);
    ~SimAnneal();
    SimAnneal(const SimAnneal &) = delete;
    SimAnneal &operator=(const SimAnneal &) = delete;
    SimAnneal(SimAnneal &&) = delete;
    SimAnneal &operator=(SimAnneal &&) = delete;
    const SimParams &effectiveParams() const { return sim_params; }
    const SearchStats &searchStats() const { return stats_; }
    static RepairResult repairConfiguration(const ublas::vector<int> &,
                                            bool known_initial_invalid = false);

    //! Invoke the desired number of annealers (threads) with the sim_params
    //! stored in the class.
    void invokeSimAnneal();

    //! Calculate system energy of a given configuration, must be exposed 
    //! publically such that the interface can recalculate system energy 
    //! for configurations storage.
    static FPType systemEnergy(const ublas::vector<int> &n_in, bool qubo=false);

    //! Return whether the given configuration is metastable.
    static bool isMetastable(const ublas::vector<int> &n_in);
    static bool validatedEnergy(const ublas::vector<int> &, FPType &);

    //! Return the charge configuration in string form.
    static std::string configToStr(const ublas::vector<int> &n_in)
    {
      std::string config_str;
      for (int chg : n_in) {
        assert(chg >= -1 && chg <= 1);
        switch(chg) {
          case -1:  config_str += "-"; break;
          case 0:   config_str += "0"; break;
          case +1:  config_str += "+"; break;
        }
      }
      return config_str;
    }


    // ACCESSORS

    //! Write simulation results of the given SimAnnealThread.
    static void storeResults(SimAnnealThread *annealer, int thread_id);

    //! Return the electron charge location results.
    AllChargeResults &chargeResults() {return charge_results;}

    //! Return the energy results.
    AllEnergyResults &energyResults() {return energy_results;}

    //! Return CPU timing results.
    //AllCPUTimes &CPUTimeingResults() {return cpu_times;}

    //! Return suggested config results.
    SuggestedResults suggestedConfigResults(bool tidy);
    const SuggestedResults &suggestedResults() const {return suggested_gs_results;}

    static FPType coulombicPotential(FPType c_1, FPType c_2, FPType eps_r, FPType lambda, FPType r);

    static FPType distance(FPType x1, FPType y1, FPType z1, FPType x2, FPType y2, FPType z2);

    //! Publically accessible simulation parameters
    static SimParams sim_params;

    // Simulation variables
    static FPType db_distance_scale;     //! convert db distances to m TODO make this configurable in user settings

  private:

    //! Initialize simulation (precomputation, setup common write-out variables,
    //! etc.
    void initialize();
    static bool evaluateConfiguration(const ublas::vector<int> &, FPType *);
    static std::recursive_mutex active_model_mutex;
    static bool model_active_;
    static std::unique_ptr<const simanneal_pair_bound::Geometry>
        repair_geometry_;
    std::recursive_mutex invocation_mutex_;
    bool invocation_active_ = false;
    SearchStats stats_;
    std::unique_ptr<refinement::Geometry> refinement_geometry_;

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
    static std::mutex result_store_mutex;

    // Runtime variables
    std::vector<std::thread> anneal_threads;  //! threads spawned


    // Write-out variables
    static AllChargeResults charge_results; // vector for storing db_charges
    static AllEnergyResults energy_results; // vector for storing config_energies
    //static AllCPUTimes cpu_times;           // vector for storing time information
    static SuggestedResults suggested_gs_results;
  };

  class SimAnnealThread
  {
  public:

    typedef boost::random::mt19937 RandEng;
    typedef boost::random::uniform_real_distribution<FPType> RandRealDist;
    typedef boost::random::uniform_int_distribution<int> RandIntDist;
    typedef boost::random::variate_generator<RandEng&, RandRealDist> RandGen;

    enum PopulationSchedulePhase{PopulationUpdateMode, PhysicalValidityCheckMode,
      PopulationUpdateFinished};

    // constructor
    SimAnnealThread(const int t_thread_id, const std::uint64_t seed);

    // destructor
    ~SimAnnealThread();

    // run simulation
    void run();

    // return the total CPU time in seconds
    /*
    double CPUTime()
    {
      // CPU time
      struct timespec curr_cpu_time;
      clockid_t thread_clock_id;
      pthread_getcpuclockid(pthread_self(), &thread_clock_id);
      clock_gettime(thread_clock_id, &curr_cpu_time);
      return (double) curr_cpu_time.tv_sec + 1e-9 * curr_cpu_time.tv_nsec;
    }
    */

    // return the physically valid ground state of this thread.
    ChargeConfigResult &suggestedConfig() {return suggested_gs;}

    int thread_id;              // the thread id of each class object
    ThreadChargeResults db_charges;       // charge configuration history
    ThreadEnergyResults config_energies;  // energy history corresponding to db_charges

  private:

    // Generate the delta in population.
    void genPopDelta(ublas::vector<int> &dn, bool &changed);

    // Start annealing.
    void anneal();

    // Perform an electron hop from one DB to another and update the energy
    // difference.
    void performHop(const int &from_ind, const int &to_ind, FPType &E_sys, 
        const FPType &E_del);

    // advance time-step
    void timeStep();

    // CALCULATIONS
    FPType systemEnergy() const;
    //FPType totalCoulombPotential(ublas::vector<int> &config) const;
    FPType hopEnergyDelta(const int &i, const int &j);

    // Return whether the current electron population is valid with an error
    // headroom to account for FPTypeing point drifts during energy updatse.
    bool populationValid() const;

    // ACCEPTANCE FUNCTIONS
    bool acceptHop(const FPType &v_diff); // acceptance function for hopping
    bool evalProb(const FPType &prob); // generate true or false based on given probaility

    // Return a random integer inclusively between min and max.
    int randInt(const int &min, const int &max);


    // VARIABLES

    // boost random number generator
    typedef simanneal_rng::Pcg32 PcgEngine;
    union RandomStorage {
      RandEng mt;
      PcgEngine pcg;
      RandomStorage() {}
      ~RandomStorage() {}
    } engines;
    bool use_pcg = false;
    FPType uniformDraw();
    RandRealDist dis01;

    // keep track of stats
    ublas::vector<int> n;       // electron configuration at the current time-step

    // other variables used for calculations
    int t=0;                        // current annealing cycle
    int t_freeze=0;                 // current v_freeze cycle
    FPType kT, v_freeze;            // current annealing temperature, freeze out potential
    FPType muzm, mupz;              // short hand for (-/0) and (0/+) charge transition levels
    ublas::vector<FPType> v_local;  // local potetial at each site

    int t_phys_validity_check=0;
    PopulationSchedulePhase pop_schedule_phase;
    int phys_valid_count;
    int phys_invalid_count;

    // keep track of the suggested config - physical valid ground state 
    // encountered by this thread.
    ChargeConfigResult suggested_gs;

    FPType E_sys;                  // energy of the system
  };
}


#endif
