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
#include <iterator>

// boost
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using namespace phys;

SimAnnealInterface::SimAnnealInterface(std::string t_in_path, 
                                       std::string t_out_path,
                                       std::string t_ext_pots_path,
                                       int t_ext_pots_step)
  : in_path(t_in_path), out_path(t_out_path), ext_pots_path(t_ext_pots_path),
    ext_pots_step(t_ext_pots_step)
{
  sqconn = new SiQADConnector(std::string("SimAnneal"), in_path, out_path);
  loadSimParams();
}

ublas::vector<FPType> SimAnnealInterface::loadExternalPotentials()
{
  Logger log(saglobal::log_level);
  bpt::ptree pt;
  bpt::read_json(ext_pots_path, pt);

  const bpt::ptree &pot_steps_arr = pt.get_child("pots");
  // iterate pots array until the desired step has been reached
  // TODO change to std::next
  if (static_cast<unsigned long>(ext_pots_step) >= pot_steps_arr.size())
    throw std::range_error("External potential step out of bounds.");
  bpt::ptree::const_iterator pots_arr_it = std::next(pot_steps_arr.begin(), 
      ext_pots_step);

  ublas::vector<FPType> v_ext;
  int db_i = 0;
  for (bpt::ptree::value_type const &v : (*pots_arr_it).second) {
    v_ext[db_i] = (v.second.get_value<FPType>()); // TODO figure out the sign
    log.debug() << "v_ext[" << db_i << "] = " << v_ext[db_i] << std::endl;
    db_i++;
  }
  return v_ext;
}

SimParams SimAnnealInterface::loadSimParams()
{
  Logger log(saglobal::log_level);

  SimParams sp;

  // grab all physical locations
  log.debug() << "Grab all physical locations..." << std::endl;
  DBLocs db_locs;
  for(auto db : *(sqconn->dbCollection())) {
    db_locs.push_back(lat_coord_to_eucl(db->n, db->m, db->l));
    log.debug() << "DB loc: x=" << db_locs.back().first
        << ", y=" << db_locs.back().second << std::endl;
  }
  sp.setDBLocs(db_locs);

  // load external voltages if relevant file has been supplied
  if (!ext_pots_path.empty()) {
    log.debug() << "Loading external potentials..." << std::endl;
    sp.v_ext = loadExternalPotentials();
  } else {
    log.debug() << "No external potentials file supplied, set to 0." << std::endl;
    for (auto &v : sp.v_ext) {
      v = 0;
    }
  }

  // VAIRABLE INITIALIZATION
  log.echo() << "Retrieving variables from SiQADConn..." << std::endl;

  // variables: physical
  sp.mu = std::stod(sqconn->getParameter("global_v0"));
  sp.epsilon_r = std::stod(sqconn->getParameter("epsilon_r"));
  sp.debye_length = 1e-9 * std::stod(sqconn->getParameter("debye_length"));

  // variables: schedule
  sp.num_instances = std::stoi(sqconn->getParameter("num_instances"));
  sp.prescale_anneal_cycles = std::stoi(sqconn->getParameter("anneal_cycles"));
  sp.preanneal_cycles = std::stoi(sqconn->getParameter("preanneal_cycles"));
  sp.hop_attempt_factor = std::stoi(sqconn->getParameter("hop_attempt_factor"));

  std::string T_schd = sqconn->getParameter("T_schedule");
  if (T_schd == "exponential") {
    sp.T_schedule = ExponentialSchedule;
  } else if (T_schd == "linear") {
    sp.T_schedule = LinearSchedule;
  } else {
    sp.T_schedule = ExponentialSchedule;
  }
  sp.T_init = std::stod(sqconn->getParameter("T_init"));
  sp.T_min = std::stod(sqconn->getParameter("T_min"));

  sp.prescale_alpha = std::stod(sqconn->getParameter("T_cycle_multiplier"));

  // variables: v_freeze related
  sp.v_freeze_init = std::stod(sqconn->getParameter("v_freeze_init"));
  sp.v_freeze_threshold = std::stod(sqconn->getParameter("v_freeze_threshold"));
  sp.v_freeze_reset = std::stod(sqconn->getParameter("v_freeze_reset"));
  sp.prescale_v_freeze_cycles = std::stoi(sqconn->getParameter("v_freeze_cycles"));
  sp.phys_validity_check_cycles = std::stoi(sqconn->getParameter("phys_validity_check_cycles"));
  sp.strategic_v_freeze_reset = sqconn->getParameter("strategic_v_freeze_reset") == "true";
  sp.reset_T_during_v_freeze_reset = sqconn->getParameter("reset_T_during_v_freeze_reset") == "true";

  // handle schedule scale factor
  sp.schedule_scale_factor = std::stod(sqconn->getParameter("schedule_scale_factor"));

  // determine result queue size, but be within the range [1,anneal_cycles]
  sp.result_queue_factor = std::stod(sqconn->getParameter("result_queue_size"));

  log.echo() << "Retrieval from SiQADConn complete." << std::endl;

  return sp;
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
      if (chg == -1) chg_ch = '-';
      else if (chg == 0) chg_ch = '0';
      else if (chg == 1) chg_ch = '+';

      elec_result_str.push_back(chg_ch);
    }
    return elec_result_str;
  };

  if (only_suggested_gs) {
    // TODO re-implement this section when needed
    throw "only_suggested_gs is currently broken.";
    /*
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
    */
  } else {
    for (auto elec_result_set : annealer->chargeResults()) {
      for (ChargeConfigResult elec_result : elec_result_set) {
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
          result.population_stable = elec_result.pop_likely_stable ?
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

int SimAnnealInterface::runSimulation(SimParams sparams)
{
  SimAnneal master_annealer(sparams);
  master_annealer.invokeSimAnneal();
  return 0;
}
