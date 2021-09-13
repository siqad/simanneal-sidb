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
  sqconn = new SiQADConnector(std::string("SimAnneal CUDA"), in_path, out_path);
  loadSimParams();
}

SimAnnealInterface::~SimAnnealInterface()
{
  delete master_annealer;
  delete sqconn;
}

ublas::vector<FPType> SimAnnealInterface::loadExternalPotentials(const int &n_dbs)
{
  Logger log(saglobal::log_level);
  bpt::ptree pt;
  bpt::read_json(ext_pots_path, pt);

  const bpt::ptree &pot_steps_arr = pt.get_child("pots");
  // iterate pots array until the desired step has been reached
  if (static_cast<unsigned long>(ext_pots_step) >= pot_steps_arr.size())
    throw std::range_error("External potential step out of bounds.");
  bpt::ptree::const_iterator pots_arr_it = std::next(pot_steps_arr.begin(), 
      ext_pots_step);

  ublas::vector<FPType> v_ext;
  v_ext.resize(n_dbs);
  int db_i = 0;
  for (bpt::ptree::value_type const &v : (*pots_arr_it).second) {
    log.debug() << "Reading v_ext[" << db_i << "]" << std::endl;
    v_ext[db_i] = (v.second.get_value<FPType>());
    log.debug() << "v_ext[" << db_i << "] = " << v_ext[db_i] << std::endl;
    db_i++;
  }
  return v_ext;
}

SimParamsCuda SimAnnealInterface::loadSimParams()
{
  Logger log(saglobal::log_level);

  SimParamsCuda sp;

  // grab all physical locations
  log.debug() << "Grab all physical locations..." << std::endl;
  std::vector<EuclCoord> db_locs;
  for(auto db : *(sqconn->dbCollection())) {
    db_locs.push_back(SimParamsCuda::latToEuclCoord(db->n, db->m, db->l));
    log.debug() << "DB loc: x=" << db_locs.back().first
        << ", y=" << db_locs.back().second << std::endl;
  }
  sp.setDBLocs(db_locs);

  // load external voltages if relevant file has been supplied
  if (!ext_pots_path.empty()) {
    log.debug() << "Loading external potentials..." << std::endl;
    sp.v_ext = loadExternalPotentials(sp.db_locs.size());
  } else {
    log.debug() << "No external potentials file supplied, set to 0." << std::endl;
    for (auto &v : sp.v_ext) {
      v = 0;
    }
  }

  // VAIRABLE INITIALIZATION
  log.echo() << "Retrieving variables from SiQADConn..." << std::endl;

  // variables: physical
  sp.mu = std::stod(sqconn->getParameter("muzm"));
  sp.eps_r = std::stod(sqconn->getParameter("eps_r"));
  sp.debye_length = std::stod(sqconn->getParameter("debye_length"));

  // variables: schedule
  sp.num_instances = std::stoi(sqconn->getParameter("num_instances"));
  sp.threads_in_instance = std::stoi(sqconn->getParameter("threads_in_instance"));
  sp.anneal_cycles = std::stoi(sqconn->getParameter("anneal_cycles"));
  //sp.preanneal_cycles = std::stoi(sqconn->getParameter("preanneal_cycles"));
  sp.hop_attempt_factor = std::stoi(sqconn->getParameter("hop_attempt_factor"));
  sp.T_e_inv_point = std::stod(sqconn->getParameter("T_e_inv_point"));

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

  // variables: v_freeze related
  sp.v_freeze_init = std::stod(sqconn->getParameter("v_freeze_init"));
  sp.v_freeze_threshold = std::stod(sqconn->getParameter("v_freeze_threshold"));
  sp.v_freeze_reset = std::stod(sqconn->getParameter("v_freeze_reset"));
  sp.v_freeze_end_point = std::stod(sqconn->getParameter("v_freeze_end_point"));
  sp.phys_validity_check_cycles = std::stoi(sqconn->getParameter("phys_validity_check_cycles"));
  sp.strategic_v_freeze_reset = sqconn->getParameter("strategic_v_freeze_reset") == "true";
  sp.reset_T_during_v_freeze_reset = sqconn->getParameter("reset_T_during_v_freeze_reset") == "true";

  // determine result queue size, but be within the range [1,anneal_cycles]
  sp.result_queue_factor = std::stod(sqconn->getParameter("result_queue_size"));

  log.echo() << "Retrieval from SiQADConn complete." << std::endl;

  return sp;
}

void SimAnnealInterface::writeSimResults(bool only_suggested_gs, bool qubo_energy)
{
  // create the vector of strings for the db locations
  std::vector<std::pair<std::string, std::string>> dbl_data(SimAnnealCuda::sim_params.db_locs.size());
  for (unsigned int i = 0; i < SimAnnealCuda::sim_params.db_locs.size(); i++) { //need the index
    dbl_data[i].first = std::to_string(SimAnnealCuda::sim_params.db_locs[i].first);
    dbl_data[i].second = std::to_string(SimAnnealCuda::sim_params.db_locs[i].second);
  }
  sqconn->setExport("db_loc", dbl_data);

  // save the results of all distributions to a map, with the vector of 
  // distribution as key and the count of occurances as value.
  struct ExportElecConfigResult
  {
    ublas::vector<int> config;  // config vector
    bool is_metastable=false;   // metastability
    FPType system_energy=-1;    // system energy
    int occ_count=0;            // occurance freq of this config
  };
  typedef std::unordered_map<std::string, ExportElecConfigResult> ElecResultMapType;
  ElecResultMapType elec_result_map;

  // process result and insert to result map
  auto process_result = [&elec_result_map, qubo_energy](
    const ublas::vector<int> &n
  ) {
    std::string n_str = SimAnnealCuda::configToStr(n);
    ExportElecConfigResult export_result;
    export_result.config = n;

    // attempt insertion
    std::pair<ElecResultMapType::iterator,bool> insert_result;
    insert_result = elec_result_map.insert({n_str, export_result});

    if (!insert_result.second) {
      // result already exists, increment counter
      insert_result.first->second.occ_count++;
    } else {
      // new result, calculate the rest of the properties
      ExportElecConfigResult &result = insert_result.first->second;
      result.is_metastable = SimAnnealCuda::isMetastable(n);
      result.system_energy = SimAnnealCuda::systemEnergy(n, qubo_energy);
      result.occ_count = 1;
    }
  };

  /*
  auto process_result = [&elec_result_map, qubo_energy](
      const ChargeConfigResult &elec_result)
  {
    if (!elec_result.isResult())
      return;

    // prepare key and val for insertion
    std::string elec_result_str = SimAnnealCuda::configToStr(elec_result.config);
    ExportElecConfigResult export_result;
    export_result.config = elec_result.config;

    // attempt insertion
    std::pair<ElecResultMapType::iterator, bool> insert_result;
    insert_result = elec_result_map.insert({elec_result_str, export_result});

    if (!insert_result.second) {
      // if insertion fails, the result already exists
      insert_result.first->second.occ_count++;
    } else {
      // if insertion succeeds, calculate the rest of the properties
      ExportElecConfigResult &result = insert_result.first->second;
      result.is_metastable = elec_result.pop_likely_stable ? 
        SimAnnealCuda::isMetastable(result.config) : false;
      // recalculate the energy for each configuration to get better accuracy
      result.system_energy = SimAnnealCuda::systemEnergy(result.config, qubo_energy);
      result.occ_count = 1;
    }
  };
  */

  // iterate through results depending on command line arguments
  for (const ublas::vector<int> &n : master_annealer->receivedResults()) {
    process_result(n);
  }

  /*
  if (only_suggested_gs) {
    for (ChargeConfigResult result : master_annealer->suggestedConfigResults(false)) {
      process_result(result);
    }
  } else {
    for (auto elec_result_set : master_annealer->chargeResults()) {
      for (ChargeConfigResult elec_result : elec_result_set) {
        process_result(elec_result);
      }
    }
  }
  */

  std::vector<std::vector<std::string>> db_dist_data;
  auto result_it = elec_result_map.cbegin();
  for (; result_it != elec_result_map.cend(); ++result_it) {
    std::vector<std::string> db_dist;
    const ExportElecConfigResult &result = result_it->second;
    db_dist.push_back(result_it->first);                      // config
    db_dist.push_back(std::to_string(result.system_energy));  // energy
    db_dist.push_back(std::to_string(result.occ_count));      // occurance freq
    db_dist.push_back(std::to_string(result.is_metastable));  // metastability
    db_dist.push_back("3");                                   // 3-state
    db_dist_data.push_back(db_dist);
  }
  sqconn->setExport("db_charge", db_dist_data);

  sqconn->writeResultsXml();
}

int SimAnnealInterface::runSimulation(SimParamsCuda &sparams)
{
  master_annealer = new SimAnnealCuda(sparams);
  master_annealer->invoke();
  return 0;
}
