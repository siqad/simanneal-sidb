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
  constexpr auto sparams = &SimAnneal::sim_params;

  // grab all physical locations (in original distance unit) (Used to be part of runSim)
  std::cout << "Grab all physical locations..." << std::endl;
  sparams->n_dbs = 0;
  for(auto db : *(sqconn->dbCollection())) {
    sparams->db_locs.push_back(std::make_pair(db->x, db->y));
    sparams->n_dbs++;
    std::cout << "DB loc: x=" << sparams->db_locs.back().first
        << ", y=" << sparams->db_locs.back().second << std::endl;
  }
  std::cout << "Free dbs, n_dbs=" << sparams->n_dbs << std::endl << std::endl;

  // exit if no dbs
  if(sparams->n_dbs == 0) {
    std::cout << "No dbs found, nothing to simulate. Exiting." << std::endl;
    std::cout << "Simulation failed, aborting" << std::endl;
    throw "No DBs found in the input file, simulation aborted.";
  }

  //Variable initialization
  std::cout << "Retrieving variables from SiQADConn..." << std::endl;
  sparams->num_threads = std::stoi(sqconn->getParameter("num_instances"));
  sparams->anneal_cycles = std::stoi(sqconn->getParameter("anneal_cycles"));
  sparams->preanneal_cycles = std::stoi(sqconn->getParameter("preanneal_cycles"));
  sparams->mu = std::stof(sqconn->getParameter("global_v0"));
  sparams->epsilon_r = std::stof(sqconn->getParameter("epsilon_r"));
  sparams->debye_length = std::stof(sqconn->getParameter("debye_length"));
  sparams->debye_length *= 1E-9; // TODO change the rest of the code to use nm instead of converting here

  std::string T_schd = sqconn->getParameter("T_schedule");
  if (T_schd == "exponential") {
    sparams->T_schedule = ExponentialSchedule;
  } else if (T_schd == "linear") {
    sparams->T_schedule = LinearSchedule;
  } else {
    sparams->T_schedule = ExponentialSchedule;
  }
  sparams->T_init = std::stof(sqconn->getParameter("T_init"));
  sparams->T_min = std::stof(sqconn->getParameter("T_min"));

  // TODO following variables should be calculated from user settings instead of hard-coded
  sparams->alpha = std::stof(sqconn->getParameter("T_cycle_multiplier"));

  sparams->v_freeze_init = std::stof(sqconn->getParameter("v_freeze_init"));
  if (sparams->v_freeze_init < 0) sparams->v_freeze_init = sparams->mu / 2;
  sparams->v_freeze_threshold = std::stof(sqconn->getParameter("v_freeze_threshold"));
  sparams->v_freeze_reset = std::stof(sqconn->getParameter("v_freeze_reset"));
  if (sparams->v_freeze_reset < 0) sparams->v_freeze_reset = sparams->mu;
  sparams->v_freeze_cycles = std::stoi(sqconn->getParameter("v_freeze_cycles"));
  sparams->phys_validity_check_cycles = std::stoi(sqconn->getParameter("phys_validity_check_cycles"));

  sparams->v_freeze_step = 
    ((sparams->v_freeze_threshold - sparams->v_freeze_init)
     / sparams->v_freeze_cycles);

  sparams->strategic_v_freeze_reset = sqconn->getParameter("strategic_v_freeze_reset") == "true";
  sparams->reset_T_during_v_freeze_reset = sqconn->getParameter("reset_T_during_v_freeze_reset") == "true";

  // handle schedule scale factor
  if (sqconn->parameterExists("schedule_scale_factor")) {
    float schd_scale_fact = std::stof(sqconn->getParameter("schedule_scale_factor"));
    float scaled_anneal_cycles = sparams->anneal_cycles * schd_scale_fact;
    float scaled_alpha = std::pow(sparams->alpha, sparams->anneal_cycles / scaled_anneal_cycles);
    float scaled_v_freeze_cycles = sparams->v_freeze_cycles * schd_scale_fact;

    std::cout << "Scaling schedule by factor " << schd_scale_fact << ":" << std::endl;
    std::cout << "Annealing cycles from " << sparams->anneal_cycles 
      << " to " << scaled_anneal_cycles << std::endl;
    std::cout << "Temperature multiplier (alpha) from " << sparams->alpha
      << " to " << scaled_alpha << std::endl;
    std::cout << "Freeze-out cycles from " << sparams->v_freeze_cycles
      << " to " << scaled_v_freeze_cycles << std::endl;

    sparams->anneal_cycles = scaled_anneal_cycles;
    sparams->alpha = scaled_alpha;
    sparams->v_freeze_cycles = scaled_v_freeze_cycles;
  }

  // determine result queue size, but be within the range [1,anneal_cycles]
  sparams->result_queue_size = sparams->anneal_cycles * std::stof(sqconn->getParameter("result_queue_size"));
  sparams->result_queue_size = std::min(sparams->result_queue_size, sparams->anneal_cycles);
  sparams->result_queue_size = std::max(sparams->result_queue_size, 1);
  std::cout << "Result queue size: " << sparams->result_queue_size << std::endl;

  std::cout << "Retrieval from SiQADConn complete." << std::endl;
}

void SimAnnealInterface::writeSimResults(bool only_suggested_gs)
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
  // TODO current only_suggested_gs processing is hacked-in and inefficient, fix
  // in the future
  typedef std::unordered_map<std::string, int> ElecResultMapType;
  ElecResultMapType elec_result_map;

  auto config_to_str = [](const ublas::vector<int> &config)
  {
    std::string elec_result_str;
    for (auto chg : config)
      elec_result_str.append(std::to_string(chg));
    return elec_result_str;
  };

  if (only_suggested_gs) {
    for (ublas::vector<int> elec_result : annealer->suggestedConfigResults()) {
      std::string elec_result_str = config_to_str(elec_result);

      // attempt insertion
      std::pair<ElecResultMapType::iterator, bool> insert_result;
      insert_result = elec_result_map.insert({elec_result_str,1});

      // if insertion fails, the result already exists. Just increment the 
      // counter within the map of that result.
      if (!insert_result.second) {
        insert_result.first->second++;
      }
    }
  } else {
    for (auto elec_result_set : annealer->chargeResults()) {
      for (ublas::vector<int> elec_result : elec_result_set) {
        std::string elec_result_str = config_to_str(elec_result);

        // attempt insertion
        std::pair<ElecResultMapType::iterator, bool> insert_result;
        insert_result = elec_result_map.insert({elec_result_str,1});

        // if insertion fails, the result already exists. Just increment the 
        // counter within the map of that result.
        if (!insert_result.second) {
          insert_result.first->second++;
        }
      }
    }
  }

  // recalculate the energy for each configuration to get better accuracy
  std::vector<std::vector<std::string>> db_dist_data;
  int i=0;
  for (auto result_it = elec_result_map.cbegin(); result_it != elec_result_map.cend(); ++result_it) {
    std::vector<std::string> db_dist;
    // config
    db_dist.push_back(result_it->first);
    // energy
    db_dist.push_back(std::to_string(annealer->systemEnergy(result_it->first, SimAnneal::sim_params.n_dbs)));
    // count
    db_dist.push_back(std::to_string(result_it->second));
    // physically valid
    db_dist.push_back(std::to_string(annealer->isPhysicallyValid(result_it->first, SimAnneal::sim_params.n_dbs)));
    db_dist_data.push_back(db_dist);
    i++;
  }
  sqconn->setExport("db_charge", db_dist_data);

  // export misc thread timing data
  unsigned int t_count = annealer->CPUTimeingResults().size();
  std::vector<std::pair<std::string, std::string>> misc_data(t_count);
  for (unsigned int i=0; i<t_count; i++) {
    misc_data[i] = std::make_pair("time_s_cpu"+std::to_string(i), 
                                  std::to_string(annealer->CPUTimeingResults().at(i)));
  }
  sqconn->setExport("misc", misc_data);

  sqconn->writeResultsXml();
}

int SimAnnealInterface::runSimulation()
{
  SimAnneal master_annealer;
  master_annealer.invokeSimAnneal();
  return 0;
}
