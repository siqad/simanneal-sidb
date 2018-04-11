// @file:     sim_anneal.h
// @author:   Samuel
// @created:  2017.08.23
// @editted:  2017.08.23 - Samuel
// @license:  GNU LGPL v3
//
// @desc:     Simulated annealing physics engine

// #include "phys_engine.h"
#include "phys_connector.h"
#include <vector>
#include <deque>
#include <tuple>
#include <memory>
#include <cmath>

#include <boost/random.hpp>

namespace constants{
  const float Q0 = 1.602E-19;
  const float PI = 3.14159;
  const float EPS0 = 8.854E-12;
  const float EPS_SURFACE = 6.35;
  const float Kb = 8.617E-5;
}

namespace phys {

  class SimAnneal
  {
  public:
    // constructor
    SimAnneal(const std::string& i_path, const std::string& o_path);

    // destructor
    ~SimAnneal() {};

    // run simulation
    bool runSim();

    // export the data through physics connector
    void exportData();


    int result_queue_size;
    std::vector<std::pair<float,float>> db_locs; // location of free dbs
    boost::circular_buffer<std::vector<int>> db_charges;
    boost::circular_buffer<float> config_energies;  // energies in line with db_charges

  private:
    // MAIN SIMULATION ROUTINE
    // initialize simulation variables
    void initVars();
    void initExpectedParams();

    // precalculate frequently used variables
    void precalc();

    // perform the simulated annealing
    void simAnneal();

    // perform an electron hop from one DB to another
    void dbHop(int from_ind, int to_ind);

    // advance time-step
    void timeStep();

    // print the current charge configuration into cout
    void printCharges();

    // CALCULATIONS
    float systemEnergy();
    float distance(float x1, float y1, float x2, float y2);
    float interElecPotential(float r);

    // ACCEPTANCE FUNCTIONS
    bool acceptPop(int db_ind);
    bool acceptHop(float v_diff); // acceptance function for hopping
    bool evalProb(float prob); // generate true or false based on given probaility

    // OTHER ACCESSORS
    int getRandDBInd(int charge);
    int chargedDBCount(int charge);


    // boost random number generator
    boost::random::mt19937 rng;

    // physics connector for interfacing with GUI
    PhysicsConnector* phys_con;

    // VARIABLES
    const int div_0 = 1E5; // arbitrary big number that represents divide by 0
    const float har_to_ev = 27.2114; // hartree to eV conversion factor
    const float db_distance_scale = 1E-10; // TODO move this to xml

    // handy constants or variables from problem file
    float Kc;           // 1 / (4 pi eps)
    float debye_length; // Silicon intrinsic Debye length in m
    float v_0;          // global bias

    // other variables used for calculations
    float kT0, kT, kT_step, v_freeze_step; // temperature, time
    int t=0, t_max, t_preanneal;      // keep track of annealing cycles
    float v_freeze;                   // freeze out potential (pushes
                                      // out population transition probability)
    std::vector<float> v_eff, v_ext, v_drive; // keep track of voltages at each
                                              // dot location
    std::vector<std::vector<float>> v_ij;     // coulombic repulsion

    // keeping track of electron configurations and other house keeping vars
    int n_dbs=-1; // number of dbs
    std::vector<std::vector<float>> db_r; // distance between all dbs
    std::vector<std::tuple<float,float,float>> fixed_charges; // location of fixed charges
    std::vector<int> curr_charges;  // electron configuration at the current time-step

  };
}
