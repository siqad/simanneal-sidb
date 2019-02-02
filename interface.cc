// @file:     interface.cc
// @author:   Samuel
// @created:  2019.02.01
// @license:  Apache License 2.0
//
// @desc:     Simulation interface which manages reads and writes with 
//            SiQADConn as well as invokes SimAnneal instances.

#include "interface.h"

// std
#include <vector>
#include <unordered_map>

// boost
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

using namespace phys;

SimAnnealInterface::SimAnnealInterface(std::string t_in_path, 
                                       std::string t_out_path)
  : in_path(t_in_path), out_path(t_out_path)
{
  sqconn = new SiQADConnector(std::string("SimAnneal"), in_path, out_path);
  loadSimParams();
}

void SimAnnealInterface::loadSimParams()
{
  // grab all physical locations (in original distance unit) (Used to be part of runSim)
  std::cout << "Grab all physical locations..." << std::endl;
  sim_params.n_dbs = 0;
  for(auto db : *(sqconn->dbCollection())) {
    sim_params.db_locs.push_back(std::make_pair(db->x, db->y));
    sim_params.n_dbs++;
    std::cout << "DB loc: x=" << sim_params.db_locs.back().first
        << ", y=" << sim_params.db_locs.back().second << std::endl;
  }
  std::cout << "Free dbs, n_dbs=" << sim_params.n_dbs << std::endl << std::endl;

  // exit if no dbs
  if(sim_params.n_dbs == 0) {
    std::cout << "No dbs found, nothing to simulate. Exiting." << std::endl;
    std::cout << "Simulation failed, aborting" << std::endl;
    throw "No DBs found in the input file, simulation aborted.";
  }

  //Variable initialization
  std::cout << "Initializing variables..." << std::endl;
  sim_params.num_threads = std::stoi(sqconn->getParameter("num_threads"));
  sim_params.anneal_cycles = std::stoi(sqconn->getParameter("anneal_cycles"));
  sim_params.mu = std::stof(sqconn->getParameter("global_v0"));
  sim_params.epsilon_r = std::stof(sqconn->getParameter("epsilon_r"));
  sim_params.debye_length = std::stof(sqconn->getParameter("debye_length"));
  sim_params.debye_length *= 1E-9; // TODO change the rest of the code to use nm / angstrom
                        //      instead of doing a conversion here.

  /* TODO add the following to sim_anneal.cc
  sim_params.kT0 = constants::Kb;
  sim_params.kT0 *= std::stof(sqconn->getParameter("min_T"));
  std::cout << "kT0 retrieved: " << std::stof(sqconn->getParameter("min_T"));
  */
  sim_params.min_T = std::stof(sqconn->getParameter("min_T"));

  sim_params.result_queue_size = std::stoi(sqconn->getParameter("result_queue_size"));
  sim_params.result_queue_size = sim_params.anneal_cycles < sim_params.result_queue_size ? sim_params.anneal_cycles : sim_params.result_queue_size;

  /* TODO add the following to sim_anneal.cc
  sim_params.Kc = 1/(4 * constants::PI * sim_accessor.epsilon_r * constants::EPS0);
  */
  sim_params.kT_step = 0.999;    // kT = Boltzmann constant (eV/K) * 298 K, NOTE kT_step arbitrary
  sim_params.v_freeze_step = 0.001;  // NOTE v_freeze_step arbitrary

  std::cout << "Variable initialization complete" << std::endl;

  /* TODO add the following to sim_anneal.cc
  sim_accessor.db_r.resize(sim_accessor.n_dbs,sim_accessor.n_dbs);
  sim_accessor.v_ext.resize(sim_accessor.n_dbs);
  sim_accessor.v_ij.resize(sim_accessor.n_dbs,sim_accessor.n_dbs);
  */
}

void SimAnnealInterface::writeSimResults()
{
  std::cout << std::endl << "*** Write Result to Output ***" << std::endl;

  // create the vector of strings for the db locations
  std::vector<std::pair<std::string, std::string>> dbl_data(sim_params.db_locs.size());
  for (unsigned int i = 0; i < sim_params.db_locs.size(); i++) { //need the index
    dbl_data[i].first = std::to_string(sim_params.db_locs[i].first);
    dbl_data[i].second = std::to_string(sim_params.db_locs[i].second);
  }
  sqconn->setExport("db_loc", dbl_data);

  // save the results of all distributions to a map, with the vector of 
  // distribution as key and the count of occurances as value.
  // TODO move typedef to header, and make typedefs for the rest of the common types
  typedef std::unordered_map<std::string, int> ElecResultMapType;
  ElecResultMapType elec_result_map;

  // TODO need new implementation of chargeStore
  for (auto elec_result_set : annealer->chargeStore) {
    for (ublas::vector<int> elec_result : elec_result_set) {
      std::string elec_result_str;
      for (auto chg : elec_result)
        elec_result_str.append(std::to_string(chg));

      // attempt insertion
      std::pair<ElecResultMapType::iterator, bool> insert_result = elec_result_map.insert({elec_result_str,1});

      // if insertion fails, the result already exists. Just increment the 
      // counter within the map of that result.
      if (!insert_result.second)
        insert_result.first->second++;
    }
  }

  // recalculate the energy for each configuration to get better accuracy
  std::vector<std::vector<std::string>> db_dist_data;
  int i=0;
  for (auto result_it = elec_result_map.cbegin(); result_it != elec_result_map.cend(); ++result_it) {
    std::vector<std::string> db_dist;
    db_dist.push_back(result_it->first);                              // config
    db_dist.push_back(std::to_string(annealer->
          systemEnergy(result_it->first)));// energy
    db_dist.push_back(std::to_string(result_it->second));             // count
    db_dist_data.push_back(db_dist);
    i++;
  }

  sqconn->setExport("db_charge", db_dist_data);
  sqconn->writeResultsXml();
}

int SimAnnealInterface::runSimulation()
{
  return 0;
}
