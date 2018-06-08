// @file:     sim_anneal.h
// @author:   Samuel
// @created:  2017.08.23
// @editted:  2017.08.23 - Samuel
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

#include <boost/random.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace constants{
  const float Q0 = 1.602E-19;
  const float PI = 3.14159;
  const float EPS0 = 8.854E-12;
  const float EPS_SURFACE = 6.35;
  const float Kb = 8.617E-5;
  const float ERFDB = 5E-10;
}

namespace phys {
  namespace ublas = boost::numeric::ublas;
  class SimAnneal
  {
  public:
    // constructor
    SimAnneal(const std::string& i_path, const std::string& o_path);

    // destructor
    ~SimAnneal() {};

    // run simulation
    void runSim();

    // export the data through physics connector
    //void exportData();

    // simmulated annealing accessor
    void simAnneal();

    //Total calculations
    float distance(const float &x1, const float &y1, const float &x2, const float &y2);
    float interElecPotential(const float &r);


    static int result_queue_size;
    static std::vector<std::pair<float,float>> db_locs; // location of free dbs
    boost::circular_buffer<ublas::vector<int>> db_charges;
    boost::circular_buffer<float> config_energies;  // energies in line with db_charges

    // keeping track of electron configurations and other house keeping vars
    static int n_dbs; // number of dbs
    int n_elec=0; // number of doubly occupied DBs
    static ublas::matrix<float> db_r;  // distance between all dbs
    ublas::vector<int> n;       // electron configuration at the current time-step
    std::vector<int> occ;       // indices of dbs, first n_elec indices are occupied
    static ublas::vector<float> v_ext; // keep track of voltages at each DB
    static ublas::matrix<float> v_ij;     // coulombic repulsion

    // other variables used for calculations
    static float kT0, kT_step, v_freeze_step;

    // handy constants or variables from problem file
    static float Kc;           // 1 / (4 pi eps)
    static float debye_length; // Silicon intrinsic Debye length in m
    static float v_0;          // global bias

    //Other variables used for CALCULATIONS
    static int t_max;                   // keep track of annealing cycles

    // VARIABLES
    //const float har_to_ev = 27.2114; // hartree to eV conversion factor
    const float db_distance_scale = 1E-10; // TODO move this to xml


  private:
    // MAIN SIMULATION ROUTINE
    // initialize simulation variables
    //void initVars();
    //void initExpectedParams();

    // precalculate frequently used variables
    //void precalc();

    // determine change in population
    ublas::vector<int> genPopDelta();

    // perform an electron hop from one DB to another
    void performHop(int from_ind, int to_ind);

    // advance time-step
    void timeStep();

    // print the current charge configuration into cout
    void printCharges();

    // CALCULATIONS
    float systemEnergy();
    float totalCoulombPotential(ublas::vector<int> config);
    float hopEnergyDelta(const int &i, const int &j);

    // ACCEPTANCE FUNCTIONS
    bool acceptPop(int db_ind);
    bool acceptHop(float v_diff); // acceptance function for hopping
    bool evalProb(float prob); // generate true or false based on given probaility

    // OTHER ACCESSORS
    int getRandOccInd(int charge);


    // boost random number generator
    boost::random::uniform_real_distribution<float> dis01;
    boost::random::mt19937 rng;

    // physics connector for interfacing with GUI
    //SiQADConnector* sqconn;

    // other variables used for calculations
                                // temperature, time
    int t=0;                   // keep track of annealing cycles
    float kT, v_freeze;                   // freeze out potential (pushes
                                      // out population transition probability)
    ublas::vector<float> v_local;
  };
}