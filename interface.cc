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
  Logger log(global::log_level);

  // grab all physical locations (in original distance unit) (Used to be part of runSim)
  log.debug() << "Grab all physical locations..." << std::endl;
  sparams->n_dbs = 0;
  for(auto db : *(sqconn->dbCollection())) {
    sparams->db_locs.push_back(lat_coord_to_eucl(db->n, db->m, db->l));
    //sparams->db_locs.push_back(std::make_pair(db->x, db->y));
    sparams->n_dbs++;
    log.debug() << "DB loc: x=" << sparams->db_locs.back().first
        << ", y=" << sparams->db_locs.back().second << std::endl;
  }
  log.debug() << "Free dbs, n_dbs=" << sparams->n_dbs << std::endl << std::endl;

  // exit if no dbs
  if(sparams->n_dbs == 0) {
    throw "No DBs found in the input file, simulation aborted.";
  }

  //Variable initialization
  log.echo() << "Retrieving variables from SiQADConn..." << std::endl;
  sparams->num_threads = std::stoi(sqconn->getParameter("num_instances"));
  sparams->anneal_cycles = std::stoi(sqconn->getParameter("anneal_cycles"));
  sparams->preanneal_cycles = std::stoi(sqconn->getParameter("preanneal_cycles"));
  sparams->hop_attempt_factor = std::stoi(sqconn->getParameter("hop_attempt_factor"));
  sparams->mu = std::stod(sqconn->getParameter("global_v0"));
  sparams->epsilon_r = std::stod(sqconn->getParameter("epsilon_r"));
  sparams->debye_length = std::stod(sqconn->getParameter("debye_length"));
  sparams->debye_length *= 1E-9; // TODO change the rest of the code to use nm instead of converting here

  std::string T_schd = sqconn->getParameter("T_schedule");
  if (T_schd == "exponential") {
    sparams->T_schedule = ExponentialSchedule;
  } else if (T_schd == "linear") {
    sparams->T_schedule = LinearSchedule;
  } else {
    sparams->T_schedule = ExponentialSchedule;
  }
  sparams->T_init = std::stod(sqconn->getParameter("T_init"));
  sparams->T_min = std::stod(sqconn->getParameter("T_min"));

  // TODO following variables should be calculated from user settings instead of hard-coded
  sparams->alpha = std::stod(sqconn->getParameter("T_cycle_multiplier"));

  sparams->v_freeze_init = std::stod(sqconn->getParameter("v_freeze_init"));
  if (sparams->v_freeze_init < 0) sparams->v_freeze_init = abs(sparams->mu) / 2;
  sparams->v_freeze_threshold = std::stod(sqconn->getParameter("v_freeze_threshold"));
  sparams->v_freeze_reset = std::stod(sqconn->getParameter("v_freeze_reset"));
  if (sparams->v_freeze_reset < 0) sparams->v_freeze_reset = abs(sparams->mu);
  sparams->v_freeze_cycles = std::stoi(sqconn->getParameter("v_freeze_cycles"));
  sparams->phys_validity_check_cycles = std::stoi(sqconn->getParameter("phys_validity_check_cycles"));

  sparams->v_freeze_step = 
    ((sparams->v_freeze_threshold - sparams->v_freeze_init)
     / sparams->v_freeze_cycles);

  sparams->strategic_v_freeze_reset = sqconn->getParameter("strategic_v_freeze_reset") == "true";
  sparams->reset_T_during_v_freeze_reset = sqconn->getParameter("reset_T_during_v_freeze_reset") == "true";

  // handle schedule scale factor
  if (sqconn->parameterExists("schedule_scale_factor")) {
    FPType schd_scale_fact = std::stod(sqconn->getParameter("schedule_scale_factor"));
    FPType scaled_anneal_cycles = sparams->anneal_cycles * schd_scale_fact;
    FPType scaled_alpha = std::pow(sparams->alpha, sparams->anneal_cycles / scaled_anneal_cycles);
    FPType scaled_v_freeze_cycles = sparams->v_freeze_cycles * schd_scale_fact;

    log.debug() << "Scaling schedule by factor " << schd_scale_fact << ":" << std::endl;
    log.debug() << "Annealing cycles from " << sparams->anneal_cycles 
      << " to " << scaled_anneal_cycles << std::endl;
    log.debug() << "Temperature multiplier (alpha) from " << sparams->alpha
      << " to " << scaled_alpha << std::endl;
    log.debug() << "Freeze-out cycles from " << sparams->v_freeze_cycles
      << " to " << scaled_v_freeze_cycles << std::endl;

    sparams->anneal_cycles = scaled_anneal_cycles;
    sparams->alpha = scaled_alpha;
    sparams->v_freeze_cycles = scaled_v_freeze_cycles;
  }

  // determine result queue size, but be within the range [1,anneal_cycles]
  sparams->result_queue_size = sparams->anneal_cycles * std::stod(sqconn->getParameter("result_queue_size"));
  sparams->result_queue_size = std::min(sparams->result_queue_size, sparams->anneal_cycles);
  sparams->result_queue_size = std::max(sparams->result_queue_size, 1);
  log.debug() << "Result queue size: " << sparams->result_queue_size << std::endl;

  log.echo() << "Retrieval from SiQADConn complete." << std::endl;
}

void SimAnnealInterface::writeSimResults(bool only_suggested_gs, bool qubo_energy)
{
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
  struct ExportElecConfigResult
  {
    ublas::vector<int> config;
    bool population_stable=false;   // the result population is physically valid
    bool locally_minimal=false;     // the result has no imminently preferred alternative configuration (if population_stable is false then this is not evaluated)
    FPType system_energy=-1;
    int occ_count=0;
  };
  typedef std::unordered_map<std::string, ExportElecConfigResult> ElecResultMapType;
  ElecResultMapType elec_result_map;

  auto config_to_str = [](const ublas::vector<int> &config)
  {
    std::string elec_result_str;
    char chg_ch;
    for (auto chg : config) {
      assert(chg >= -1 && chg <= 1);
      if (chg == -1) chg_ch = '+';
      else if (chg == 0) chg_ch = '0';
      else if (chg == 1) chg_ch = '-';

      elec_result_str.push_back(chg_ch);
    }
    return elec_result_str;
  };

  if (only_suggested_gs) {
    for (ublas::vector<int> elec_result : annealer->suggestedConfigResults()) {
      std::string elec_result_str = config_to_str(elec_result); // key
      ExportElecConfigResult export_result;                     // value
      export_result.config = elec_result;

      // attempt insertion
      std::pair<ElecResultMapType::iterator, bool> insert_result;
      insert_result = elec_result_map.insert({elec_result_str,export_result});

      if (!insert_result.second) {
        // if insertion fails, the result already exists. Just increment the 
        // occ_counter within the map of that result.
        insert_result.first->second.occ_count++;
      } else {
        // if insertion succeeds, then this is the first insertion and the 
        // rest of the result properties must be determined.
        ExportElecConfigResult &result = insert_result.first->second;
        result.population_stable = SimAnneal::populationValidity(result.config);
        result.locally_minimal = result.population_stable ? 
          SimAnneal::locallyMinimal(result.config) : false;
        result.system_energy = SimAnneal::systemEnergy(result.config, qubo_energy);
      }
    }
  } else {
    for (auto elec_result_set : annealer->chargeResults()) {
      for (ElecChargeConfigResult elec_result : elec_result_set) {
        if (!elec_result.isResult())
          continue;
        std::string elec_result_str = config_to_str(elec_result.config);  // key
        ExportElecConfigResult export_result;                             // val
        export_result.config = elec_result.config;

        // attempt insertion
        std::pair<ElecResultMapType::iterator, bool> insert_result;
        insert_result = elec_result_map.insert({elec_result_str,export_result});

        if (!insert_result.second) {
          // if insertion fails, the result already exists. Just increment the 
          // counter within the map of that result.
          insert_result.first->second.occ_count++;
        } else {
          // if insertion succeeds, then this is the first insertion and the 
          // rest of the result properties must be determined.
          ExportElecConfigResult &result = insert_result.first->second;
          result.population_stable = elec_result.population_possibly_stable ?
            SimAnneal::populationValidity(result.config) : false;
          result.locally_minimal = result.population_stable ?
            SimAnneal::locallyMinimal(result.config) : false;
          result.system_energy = SimAnneal::systemEnergy(result.config, qubo_energy);
        }
      }
    }
  }

  // recalculate the energy for each configuration to get better accuracy
  std::vector<std::vector<std::string>> db_dist_data;
  auto result_it = elec_result_map.cbegin();
  for (; result_it != elec_result_map.cend(); ++result_it) {
    std::vector<std::string> db_dist;
    const ExportElecConfigResult &result = result_it->second;
    // config
    db_dist.push_back(result_it->first);
    // energy
    db_dist.push_back(std::to_string(result.system_energy));
    // count
    db_dist.push_back(std::to_string(result.occ_count));
    // physically valid
    db_dist.push_back(std::to_string(result.population_stable && result.locally_minimal));
    // 3 state export
    db_dist.push_back("3");
    db_dist_data.push_back(db_dist);
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
