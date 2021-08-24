// @file:     simanneal_cuda.h
// @author:   Samuel
// @created:  2021.07.27
// @license:  Apache License 2.0
//
// @desc:     Simulated annealing physics engine written in CUDA

#ifndef _PHYS_SIMANNEAL_CUDA_H_
#define _PHYS_SIMANNEAL_CUDA_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <iostream>
#include <math.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include "constants.h"
#include "global.h"

namespace phys {
  namespace ublas = boost::numeric::ublas;

  // enums
  enum TemperatureSchedule{LinearSchedule, ExponentialSchedule};

  typedef std::pair<FPType,FPType> EuclCoord;
  typedef std::vector<int> LatCoord;

  struct SimParamsCuda
  {
    SimParamsCuda() {};

    void setDBLocs(const std::vector<EuclCoord> &);

    void setDBLocs(const std::vector<LatCoord> &);

    static EuclCoord latToEuclCoord(const int &n, const int &m, const int &l);

    // used for alpha and v_freeze_cycles calculation
    int anneal_cycles=10000;      // Total number of annealing cycles
    FPType T_e_inv_point=0.09995; // Where in schedule does T = 1/e * T_0
    FPType v_freeze_end_point=0.4;// Where in schedule does v_freeze stop growing

    // runtime params
    int num_instances=-1;         // Number of threads to spawn
    float result_queue_factor=.1; // Number of results to store (per thread)
    int result_queue_size;        // Number of results to store (per thread)
    int hop_attempt_factor=5;     // total hop attempt = hop_attempt_factor * (num_occ - num_vac)

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
    bool strategic_v_freeze_reset=true;
    bool reset_T_during_v_freeze_reset=true;

    // calculated params (from annealing params)
    FPType Kc;                   // 1 / (4 pi eps)
    FPType kT_min;               // Kb * T_min
    FPType v_freeze_step;        // (v_freeze_threshold - v_freeze_init) / v_freeze_cycles

    // physics params
    FPType mu=-0.25;             // Global Fermi level (eV)
    FPType eps_r=5.6;        // Relative premittivity on the surface
    FPType debye_length=5.0;     // Debye Length (nm)
    int n_dbs;                   // Number of DBs in the simulation
    std::vector<std::pair<FPType,FPType>> db_locs;  // Location of DBs
    ublas::matrix<FPType> db_r;  // Matrix of distances between all DBs
    ublas::matrix<FPType> v_ij;  // Matrix of coulombic repulsion between occupied DBs
    ublas::vector<FPType> v_ext; // External potential influences
  };

  class SimAnnealCuda
  {
  public:

    //! Constructor taking the simulation parameters.
    SimAnnealCuda(SimParamsCuda &sparams);

    //! Invoke the CUDA simulated annealing.
    void invoke();

    //! Calculate system energy of a given configuration, must be exposed 
    //! publically such that the interface can recalculate system energy 
    //! for configurations storage.
    static FPType systemEnergy(const ublas::vector<int> &n_in, bool qubo=false);

    //! Return whether the given configuration is metastable.
    static bool isMetastable(const ublas::vector<int> &n_in);

    static SimParamsCuda sim_params;

  private:

    //! Initialize simulation.
    void initialize();

    //! Calculate the Euclidean distance between the i th and j th DBs in the 
    //! db_locs array.
    FPType distance(const int &i, const int &j);

    //! Calculate the potential between two given point charges.
    FPType interElecPotential(const FPType &r);

    //! Return the energy difference for a configuration if an electron hopped
    //! from site i to site j.
    static FPType hopEnergyDelta(ublas::vector<int> n_in, const int &from_ind,
        const int &to_ind);

    // Simulation variables
    static FPType db_distance_scale;     //! convert db distances to m TODO make this configurable in user settings
  };

  class TestAdd {
  public:

    TestAdd() {}

    void runAdd();

  };

}

#endif
