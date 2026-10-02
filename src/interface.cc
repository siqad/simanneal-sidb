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
#include <iomanip>
#include <sstream>
#include <limits>

// boost
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using namespace phys;

SimAnnealInterface::SimAnnealInterface(std::string t_in_path, 
                                       std::string t_out_path,
                                       std::string t_ext_pots_path,
                                       int t_ext_pots_step,
                                       bool verbose)
  : in_path(t_in_path), out_path(t_out_path), ext_pots_path(t_ext_pots_path),
    ext_pots_step(t_ext_pots_step)
{
  sqconn.reset(new SiQADConnector(std::string("SimAnneal"), in_path, out_path, verbose));
}

SimAnnealInterface::~SimAnnealInterface()
{
  // Members retain the active model through export and release it on failure.
}

ublas::vector<FPType> SimAnnealInterface::loadExternalPotentials(const int &n_dbs)
{
  Logger log(saglobal::log_level);
  bpt::ptree pt;
  bpt::read_json(ext_pots_path, pt);

  const bpt::ptree &pot_steps_arr = pt.get_child("pots");
  // iterate pots array until the desired step has been reached
  if (ext_pots_step < 0 || static_cast<unsigned long>(ext_pots_step) >= pot_steps_arr.size())
    throw std::range_error("External potential step out of bounds.");
  bpt::ptree::const_iterator pots_arr_it = std::next(pot_steps_arr.begin(), 
      ext_pots_step);

  if ((*pots_arr_it).second.size() != static_cast<std::size_t>(n_dbs))
    throw std::invalid_argument("External potential count must equal the number of sites");
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

SimParams SimAnnealInterface::loadSimParams()
{
  Logger log(saglobal::log_level);

  SimParams sp;

  // find the lattice layer in the layer list and get the lattice vector
  bool valid_lat_vec_found = false;
  for (auto layer : sqconn->getLayers()) {
    log.debug() << "Layer name: " << layer.name << ", layer type: " << layer.type << std::endl;
    if (layer.type.compare("Lattice") == 0) {
      if (!layer.lat_vec.isValid()) {
        continue;
      }
      valid_lat_vec_found = true;
      sp.lat_vec.a1 = layer.lat_vec.a1;
      sp.lat_vec.a2 = layer.lat_vec.a2;
      sp.lat_vec.atoms = layer.lat_vec.atoms;
      log.debug() << "Lattice vector found: " << layer.lat_vec.name << std::endl;
      log.debug() << "Lattice vector: a1=(" << sp.lat_vec.a1.first << "," << sp.lat_vec.a1.second << ")"
          << ", a2=(" << sp.lat_vec.a2.first << "," << sp.lat_vec.a2.second << ")" << std::endl;
      break;
    }
  }
  if (!valid_lat_vec_found || !sp.lat_vec.isValid()) {
    throw std::runtime_error("No valid lattice vector found.");
  }

  // grab all physical locations
  log.debug() << "Grab all physical locations..." << std::endl;
  std::vector<EuclCoord> db_locs;
  for(auto db : *(sqconn->dbCollection())) {
    db_locs.push_back(SimParams::latToEuclCoord(db->n, db->m, db->l, sp.lat_vec));
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

  // grab defects
  log.debug() << "Grab all defects..." << std::endl;
  std::vector<EuclCoord3d> defect_locs;
  std::vector<FPType> fixed_charges;
  std::vector<FPType> fixed_charge_eps_rs;
  std::vector<FPType> fixed_charge_lambdas;
  for (auto defect : *(sqconn->defectCollection())) {
    if (defect->has_eucl) {
      log.debug() << "**** HAS EUCL ****" << std::endl;
      defect_locs.push_back(EuclCoord3d(defect->x, defect->y, defect->z));
    } else if (defect->has_lat_coord) {
      log.debug() << "**** HAS LATCOORD ****" << std::endl;
      // TODO find center point and calculate that x y z
      EuclCoord anchor_tl = SimParams::latToEuclCoord(defect->n, defect->m, defect->l, sp.lat_vec);
      log.debug() << "tl -- n=" << defect->n << ", m=" << defect->m << ", l=" << defect->l << std::endl;
      log.debug() << "tl -- x=" << anchor_tl.first << ", y=" << anchor_tl.second << std::endl;
      if (defect->w == 1 && defect->h == 1) {
        defect_locs.push_back(EuclCoord3d(anchor_tl.first, anchor_tl.second, 0));
      } else {
        int n_br = defect->n + defect->w;
        int m_br = defect->m + (int) floor((defect->h - defect->l)/ 2);
        int l_br = (defect->h - defect->l) % 2;
        log.debug() << "br -- n=" << n_br << ", m=" << m_br << ", l=" << l_br << std::endl;
        EuclCoord anchor_br = SimParams::latToEuclCoord(n_br, m_br, l_br, sp.lat_vec);
        log.debug() << "br -- x=" << anchor_br.first << ", y=" << anchor_br.second << std::endl;
        FPType x_mean = (anchor_br.first + anchor_tl.first) / 2;
        FPType y_mean = (anchor_br.second + anchor_tl.second)  / 2;
        log.debug() << "mean -- x=" << x_mean << ", y=" << y_mean << std::endl;
        defect_locs.push_back(EuclCoord3d(x_mean, y_mean, 0));
      }
    } else {
      log.debug() << "No location info provided for one of the defects, skipping" << std::endl;
      continue;
    }
    fixed_charges.push_back(defect->charge);
    fixed_charge_eps_rs.push_back(defect->eps_r);
    fixed_charge_lambdas.push_back(defect->lambda_tf);

    log.debug() << "Defect loc: x=" << defect_locs.back().x 
        << ", y=" << defect_locs.back().y
        << ", z=" << defect_locs.back().z
        << ", charge=" << fixed_charges.back()
        << std::endl;
  }
  sp.setFixedCharges(defect_locs, fixed_charges, fixed_charge_eps_rs, fixed_charge_lambdas);

  // VAIRABLE INITIALIZATION
  log.echo() << "Retrieving variables from SiQADConn..." << std::endl;

  // variables: physical
  sp.mu = std::stod(sqconn->getParameter("muzm"));
  sp.eps_r = std::stod(sqconn->getParameter("eps_r"));
  sp.debye_length = std::stod(sqconn->getParameter("debye_length"));

  // variables: schedule
  sp.num_instances = std::stoi(sqconn->getParameter("num_instances"));
  const auto workers = sqconn->getParameter("num_workers");
  if (!workers.empty()) sp.num_workers = std::stoi(workers);
  const auto population_backend = sqconn->getParameter("population_backend");
  if (population_backend == "portable") sp.population_backend = PopulationBackend::Portable;
  else if (population_backend == "accelerate") sp.population_backend = PopulationBackend::Accelerate;
  else if (population_backend == "openblas") sp.population_backend = PopulationBackend::OpenBLAS;
  else if (population_backend == "openblas_symmetric") sp.population_backend = PopulationBackend::OpenBLASSymmetric;
  else if (!population_backend.empty() && population_backend != "auto")
    throw std::invalid_argument("Unknown population_backend: " + population_backend);
  const auto profile = sqconn->getParameter("search_profile");
  if (profile == "optimized") sp.search_profile = SearchProfile::Optimized;
  else if (!profile.empty() && profile != "legacy")
    throw std::invalid_argument("Unknown search_profile: " + profile);
  const auto rng = sqconn->getParameter("random_backend");
  if (rng == "mt") sp.random_backend = RandomBackend::MT;
  else if (rng == "pcg32") sp.random_backend = RandomBackend::PCG32;
  else if (!rng.empty() && rng != "auto")
    throw std::invalid_argument("Unknown random_backend: " + rng);
  auto feature = [&](const std::string &name, FeatureSetting &value) {
    const auto text = sqconn->getParameter(name);
    if (text.empty() || text == "profile") return;
    if (text == "true" || text == "1") value = FeatureSetting::Enabled;
    else if (text == "false" || text == "0") value = FeatureSetting::Disabled;
    else throw std::invalid_argument(name + " must be profile, true, or false");
  };
  feature("repair", sp.repair);
  feature("singleton_shortcut", sp.singleton_shortcut);
  auto boolean = [&](const std::string &name, bool &value) {
    const auto text = sqconn->getParameter(name);
    if (text.empty()) return;
    if (text == "true" || text == "1") value = true;
    else if (text == "false" || text == "0") value = false;
    else throw std::invalid_argument(name + " must be true or false");
  };
  boolean("probability_shortcuts", sp.probability_shortcuts);
  boolean("transient_domain_mask", sp.transient_domain_mask);
  const auto refinement = sqconn->getParameter("refinement");
  if (refinement == "k6") sp.refinement_options.mode = refinement::Mode::K6;
  else if (refinement == "k10") sp.refinement_options.mode = refinement::Mode::K10;
  else if (refinement == "shared") sp.refinement_options.mode = refinement::Mode::SharedK10;
  else if (!refinement.empty() && refinement != "none")
    throw std::invalid_argument("Unknown refinement: " + refinement);
  auto integer = [&](const std::string &name, int &value) {
    const auto text = sqconn->getParameter(name);
    if (text.empty()) return;
    std::size_t used = 0;
    value = std::stoi(text, &used);
    if (used != text.size()) throw std::invalid_argument(name + " must be an integer");
  };
  integer("refinement_candidates", sp.refinement_options.candidates);
  integer("refinement_rounds", sp.refinement_options.rounds);
  integer("refinement_trials", sp.refinement_options.trials);
  const auto history = sqconn->getParameter("record_history");
  sp.record_history = history == "true" || history == "1";
  const auto seed = sqconn->getParameter("random_seed");
  if (sqconn->parameterExists("random_seed") && seed != "random") {
    sp.random_seed = SimParams::parseRandomSeed(seed);
    sp.deterministic_seed = true;
  }
  sp.anneal_cycles = std::stoi(sqconn->getParameter("anneal_cycles"));
  //sp.preanneal_cycles = std::stoi(sqconn->getParameter("preanneal_cycles"));
  sp.hop_attempt_factor = std::stoi(sqconn->getParameter("hop_attempt_factor"));
  // Optional for backwards compatibility with existing problem XML files.
  const auto hop_policy = sqconn->getParameter("hop_selection");
  if (hop_policy == "local_uniform") sp.hop_selection = LocalUniformHop;
  else if (hop_policy == "local_distance") sp.hop_selection = LocalDistanceHop;
  else if (hop_policy == "local_radius") sp.hop_selection = LocalRadiusHop;
  else if (!hop_policy.empty() && hop_policy != "uniform")
    throw std::invalid_argument("Unknown hop_selection: " + hop_policy);
  const auto hop_k = sqconn->getParameter("hop_neighbors");
  const auto hop_length = sqconn->getParameter("hop_length_nm");
  const auto hop_radius = sqconn->getParameter("hop_radius_nm");
  const auto hop_global = sqconn->getParameter("hop_global_probability");
  if (!hop_k.empty()) sp.hop_neighbors = std::stoi(hop_k);
  if (!hop_length.empty()) sp.hop_length_nm = std::stod(hop_length);
  if (!hop_radius.empty()) sp.hop_radius_nm = std::stod(hop_radius);
  if (!hop_global.empty()) sp.hop_global_probability = std::stod(hop_global);
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
  if (!master_annealer) throw std::logic_error("Run simulation before exporting results");
  // create the vector of strings for the db locations
  std::vector<std::pair<std::string, std::string>> dbl_data(SimAnneal::sim_params.db_locs.size());
  for (unsigned int i = 0; i < SimAnneal::sim_params.db_locs.size(); i++) { //need the index
    dbl_data[i].first = std::to_string(SimAnneal::sim_params.db_locs[i].first);
    dbl_data[i].second = std::to_string(SimAnneal::sim_params.db_locs[i].second);
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
      const ChargeConfigResult &elec_result)
  {
    if (!elec_result.isResult())
      return;

    // prepare key and val for insertion
    std::string elec_result_str = SimAnneal::configToStr(elec_result.config);
    auto existing = elec_result_map.find(elec_result_str);
    if (existing != elec_result_map.end()) {
      ++existing->second.occ_count;
      return;
    }
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
        SimAnneal::isMetastable(result.config) : false;
      // recalculate the energy for each configuration to get better accuracy
      result.system_energy = SimAnneal::systemEnergy(result.config, qubo_energy);
      result.occ_count = 1;
    }
  };

  // iterate through results depending on command line arguments
  for (const ChargeConfigResult &result : master_annealer->suggestedResults()) {
    process_result(result);
  }
  if (!only_suggested_gs) {
    for (const auto &elec_result_set : master_annealer->chargeResults()) {
      for (const ChargeConfigResult &elec_result : elec_result_set) {
        process_result(elec_result);
      }
    }
  }

  std::vector<std::vector<std::string>> db_dist_data;
  auto result_it = elec_result_map.cbegin();
  for (; result_it != elec_result_map.cend(); ++result_it) {
    std::vector<std::string> db_dist;
    const ExportElecConfigResult &result = result_it->second;
    db_dist.push_back(result_it->first);                      // config
    std::ostringstream energy;
    energy << std::setprecision(std::numeric_limits<FPType>::max_digits10) << result.system_energy;
    db_dist.push_back(energy.str());                         // lossless FP64 energy
    db_dist.push_back(std::to_string(result.occ_count));      // occurance freq
    db_dist.push_back(std::to_string(result.is_metastable));  // metastability
    db_dist.push_back("3");                                   // 3-state
    db_dist_data.push_back(db_dist);
  }
  sqconn->setExport("db_charge", db_dist_data);

  const auto &effective = master_annealer->effectiveParams();
  const auto &stats = master_annealer->searchStats();
  std::string backend = "portable";
  switch (effective.population_backend) {
    case PopulationBackend::Portable: break;
    case PopulationBackend::Accelerate: backend="accelerate"; break;
    case PopulationBackend::OpenBLAS: backend="openblas"; break;
    case PopulationBackend::OpenBLASSymmetric: backend="openblas_symmetric"; break;
    case PopulationBackend::Auto:
#ifdef SIMANNEAL_HAVE_ACCELERATE
      backend="accelerate";
#endif
      break;
  }
  const auto mode = effective.refinement_options.mode;
  const std::string refinement_mode = mode==refinement::Mode::Disabled ? "none" :
      mode==refinement::Mode::K6 ? "k6" : mode==refinement::Mode::K10 ? "k10" : "shared";
  const auto boolean = [](bool value) {return value ? "true" : "false";};
  std::vector<std::pair<std::string,std::string>> metadata{
    {"search_profile", effective.search_profile==SearchProfile::Legacy ? "legacy" : "optimized"},
    {"random_backend", effective.random_backend==RandomBackend::PCG32 ? "pcg32" : "mt"},
    {"population_backend", backend},
    {"probability_shortcuts", boolean(effective.probability_shortcuts)},
    {"repair", boolean(effective.repair_enabled)},
    {"transient_domain_mask", boolean(effective.transient_domain_mask)},
    {"singleton_shortcut", boolean(effective.singleton_enabled)},
    {"singleton_used", boolean(stats.singleton_used)},
    {"requested_restarts", std::to_string(effective.num_instances)},
    {"executed_restarts", std::to_string(stats.executed_restarts)},
    {"active_workers", std::to_string(effective.num_workers)},
    {"repair_attempts", std::to_string(stats.repair_attempts)},
    {"repair_budget_exhaustions", std::to_string(stats.repair_budget_exhaustions)},
    {"refinement", refinement_mode},
    {"refinement_candidates", std::to_string(effective.refinement_options.candidates)},
    {"refinement_rounds", std::to_string(effective.refinement_options.rounds)},
    {"refinement_trials", std::to_string(effective.refinement_options.trials)},
    {"refinement_center_offset", std::to_string(stats.refinement_center_offset)},
    {"refinement_selected", std::to_string(stats.refinement.selected)},
    {"refinement_improvements", std::to_string(stats.refinement.improvements)},
    {"refinement_budget_exhausted", boolean(stats.refinement.budget_exhausted)},
    {"refinement_geometry_skipped", boolean(stats.refinement.geometry_skipped)},
    {"refinement_geometry_bytes", std::to_string(stats.refinement.geometry_bytes)}
  };
  std::size_t refinement_results = 0;
  for (const auto &result : master_annealer->suggestedResults())
    refinement_results += result.refinement_result;
  metadata.emplace_back("refinement_result_count", std::to_string(refinement_results));
  metadata.emplace_back("occurrence_semantics", "exported records; restart results plus optional refinement result and diagnostic history");
  sqconn->setExport("misc", metadata);

  sqconn->writeResultsXml();
}

int SimAnnealInterface::runSimulation(SimParams sparams)
{
  master_annealer.reset();
  master_annealer.reset(new SimAnneal(sparams));
  master_annealer->invokeSimAnneal();
  return 0;
}
