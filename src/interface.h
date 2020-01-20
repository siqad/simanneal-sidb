// @file:     interface.h
// @author:   Samuel
// @created:  2019.02.01
// @license:  Apache License 2.0
//
// @desc:     Simulation interface which manages reads and writes with 
//            SiQADConn as well as invokes SimAnneal instances.

#include "siqadconn.h"
#include "simanneal.h"
#include <string>

namespace phys {

  namespace bpt = boost::property_tree;

  class helper;

  class SimAnnealInterface
  {
  public:
    //! Constructure for SimAnnealInterface. Set defer_var_loading to true if
    //! you don't want simulation parameters to be loaded immediately from
    //! SiQADConn.
    SimAnnealInterface(std::string t_in_path, std::string t_out_path, 
        std::string t_ext_pots_path, int t_ext_pots_step);

    //! Read external potentials.
    ublas::vector<FPType> loadExternalPotentials();

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

    //! Convert lattice coordinates (n, m, l) to a pair of Euclidean coordinates 
    //! in angstrom.
    EuclCoord2D lat_coord_to_eucl(int n, int m, int l)
    {
      FPType x = n * constants::lat_a;
      FPType y = m * constants::lat_b + l * constants::lat_c;
      return std::make_pair(x, y);
    }

    // Instances
    SiQADConnector *sqconn=nullptr;
    SimAnneal *annealer=nullptr;

    // variables
    std::string in_path;
    std::string out_path;
    std::string ext_pots_path;
    int ext_pots_step;

  };

  class helper {
    template <typename T = std::string> 
    T element_at_checked(bpt::ptree const& pt, std::string name, size_t n) {
        auto r = pt.get_child(name).equal_range("");

        for (; r.first != r.second && n; --n) ++r.first;

        if (n || r.first==r.second)
            throw std::range_error("index out of bounds");

        return r.first->second.get_value<T>();
    }
  };
}
