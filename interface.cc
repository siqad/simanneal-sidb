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
  SimAnneal::sim_params.n_dbs = 0;
  for(auto db : *(sqconn->dbCollection())) {
    SimAnneal::sim_params.db_locs.push_back(std::make_pair(db->x, db->y));
    SimAnneal::sim_params.n_dbs++;
    std::cout << "DB loc: x=" << SimAnneal::sim_params.db_locs.back().first
        << ", y=" << SimAnneal::sim_params.db_locs.back().second << std::endl;
  }
  std::cout << "Free dbs, n_dbs=" << SimAnneal::sim_params.n_dbs << std::endl << std::endl;

  // exit if no dbs
  if(SimAnneal::sim_params.n_dbs == 0) {
    std::cout << "No dbs found, nothing to simulate. Exiting." << std::endl;
    std::cout << "Simulation failed, aborting" << std::endl;
    throw "No DBs found in the input file, simulation aborted.";
  }

  //Variable initialization
  std::cout << "Retrieving variables from SiQADConn..." << std::endl;
  SimAnneal::sim_params.num_threads = std::stoi(sqconn->getParameter("num_threads"));
  SimAnneal::sim_params.anneal_cycles = std::stoi(sqconn->getParameter("anneal_cycles"));
  SimAnneal::sim_params.preanneal_cycles = std::stoi(sqconn->getParameter("preanneal_cycles"));
  SimAnneal::sim_params.mu = std::stof(sqconn->getParameter("global_v0"));
  SimAnneal::sim_params.epsilon_r = std::stof(sqconn->getParameter("epsilon_r"));
  SimAnneal::sim_params.debye_length = std::stof(sqconn->getParameter("debye_length"));
  SimAnneal::sim_params.debye_length *= 1E-9; // TODO change the rest of the code to use nm instead of converting here

  SimAnneal::sim_params.T_init = std::stof(sqconn->getParameter("T_init"));
  SimAnneal::sim_params.T_min = std::stof(sqconn->getParameter("T_min"));

  SimAnneal::sim_params.result_queue_size = std::stoi(sqconn->getParameter("result_queue_size"));
  SimAnneal::sim_params.result_queue_size = SimAnneal::sim_params.anneal_cycles < SimAnneal::sim_params.result_queue_size ? SimAnneal::sim_params.anneal_cycles : SimAnneal::sim_params.result_queue_size;

  // TODO following variables should be calculated from user settings instead of hard-coded
  SimAnneal::sim_params.alpha = std::stof(sqconn->getParameter("T_cycle_multiplier"));
  SimAnneal::sim_params.v_freeze_step = 0.001;  // NOTE v_freeze_step arbitrary

  std::cout << "Retrieval from SiQADConn complete." << std::endl;
}

void SimAnnealInterface::writeSimResults()
{
  std::cout << std::endl << "*** Write Result to Output ***" << std::endl;

  // create the vector of strings for the db locations
  std::vector<std::pair<std::string, std::string>> dbl_data(SimAnneal::sim_params.db_locs.size());
  for (unsigned int i = 0; i < SimAnneal::sim_params.db_locs.size(); i++) { //need the index
    dbl_data[i].first = std::to_string(SimAnneal::sim_params.db_locs[i].first);
    dbl_data[i].second = std::to_string(SimAnneal::sim_params.db_locs[i].second);
  }
  sqconn->setExport("db_loc", dbl_data);

  // save the results of all distributions to a map, with the vector of 
  // distribution as key and the count of occurances as value.
  // TODO move typedef to header, and make typedefs for the rest of the common types
  typedef std::unordered_map<std::string, int> ElecResultMapType;
  ElecResultMapType elec_result_map;

  for (auto elec_result_set : annealer->chargeResults()) {
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
    db_dist.push_back(std::to_string(annealer->systemEnergy(result_it->first, SimAnneal::sim_params.n_dbs)));// energy
    db_dist.push_back(std::to_string(result_it->second));             // count
    db_dist_data.push_back(db_dist);
    i++;
  }

  sqconn->setExport("db_charge", db_dist_data);
  sqconn->writeResultsXml();
}

int SimAnnealInterface::runSimulation()
{
  SimAnneal master_annealer;
  master_annealer.invokeSimAnneal();
  return 0;
}
