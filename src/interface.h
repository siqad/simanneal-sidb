// @file:     interface.h
// @author:   Samuel
// @created:  2019.02.01
// @license:  Apache License 2.0
//
// @desc:     Simulation interface which manages reads and writes with 
//            SiQADConn as well as invokes SimAnneal instances.

#include "libs/siqadconn/src/siqadconn.h"
#include "simanneal.h"
#include <string>

namespace phys {

  namespace bpt = boost::property_tree;

  class SimAnnealInterface
  {
  public:
    //! Constructure for SimAnnealInterface. Set defer_var_loading to true if
    //! you don't want simulation parameters to be loaded immediately from
    //! SiQADConn.
    SimAnnealInterface(std::string t_in_path, std::string t_out_path, 
        std::string t_ext_pots_path, int t_ext_pots_step, bool verbose=false);

    ~SimAnnealInterface();

    //! Read external potentials.
    ublas::vector<FPType> loadExternalPotentials(const int &n_dbs);

    //! Prepare simulation variables.
    SimParams loadSimParams();

    //! Write the simulation results to output file. The only_suggested_gs flag
    //! instructs the function to only export the single suggested ground state 
    //! result from each thread rather than exporting all results in the result
    //! queue.
    void writeSimResults(bool only_suggested_gs, bool qubo_energy);


    //! Run the simulation, returns 0 if simulation was successful.
    int runSimulation(SimParams sparams);

  private:

    // Instances
    SiQADConnector *sqconn=nullptr;
    SimAnneal *master_annealer=nullptr;

    // variables
    std::string in_path;
    std::string out_path;
    std::string ext_pots_path;
    int ext_pots_step;

  };
}
