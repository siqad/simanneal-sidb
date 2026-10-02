// @file:     sim_anneal.cc
// @author:   Samuel
// @created:  2017.08.23
// @editted:  2019.02.01
// @license:  Apache License 2.0
//
// @desc:     Simulated annealing physics engine

#include "simanneal.h"
#ifdef SIMANNEAL_HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif
#include <ctime>
#include <algorithm>
#include <unordered_set>
#include <atomic>
#include <exception>
#include <stdexcept>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

// thread CPU time for Linux
//#include <pthread.h>
//#include <time.h>

saglobal::TimeKeeper *saglobal::TimeKeeper::time_keeper=nullptr;
int saglobal::log_level = Logger::WRN;

using namespace phys;

// static variables
SimParams SimAnneal::sim_params;
std::mutex SimAnneal::result_store_mutex;
FPType SimAnneal::db_distance_scale = 1E-10;
AllChargeResults SimAnneal::charge_results;
AllEnergyResults SimAnneal::energy_results;
//AllCPUTimes SimAnneal::cpu_times;
SuggestedResults SimAnneal::suggested_gs_results;

// alias for the commonly used sim_params static variable
constexpr auto sparams = &SimAnneal::sim_params;


// SimParams implementation

void SimParams::setDBLocs(const std::vector<EuclCoord> &t_db_locs)
{
  db_locs = t_db_locs;
  if (db_locs.size() == 0) {
    throw "There must be 1 or more DBs when setting DBs for SimParams.";
  }
  n_dbs = db_locs.size();
  db_r.resize(n_dbs, n_dbs);
  v_ij.resize(n_dbs, n_dbs);
  v_ext.resize(n_dbs);
  v_fc.resize(n_dbs);
}

void SimParams::setDBLocs(const std::vector<LatCoord> &t_db_locs, const phys::LatticeVector &lat_vec)
{
  std::vector<EuclCoord> db_locs;
  for (LatCoord lat_coord : t_db_locs) {
    assert(lat_coord.size() == 3);
    db_locs.push_back(latToEuclCoord(lat_coord[0], lat_coord[1], lat_coord[2], lat_vec));
  }
  setDBLocs(db_locs);
}

EuclCoord SimParams::latToEuclCoord(const int &n, const int &m, const int &l, const phys::LatticeVector &lat_vec)
{
  FPType x = n * (lat_vec.a1.first + lat_vec.a2.first) + lat_vec.atoms[l].first;
  FPType y = m * (lat_vec.a1.second + lat_vec.a2.second) + lat_vec.atoms[l].second;
  return std::make_pair(x, y);
}

void SimParams::setFixedCharges(const std::vector<EuclCoord3d> &t_fc_locs, 
  const std::vector<FPType> &t_fcs, const std::vector<FPType> &t_fc_eps_rs,
  const std::vector<FPType> &t_fc_lambdas)
{
  // fold fixed charge defect effects into v_fc
  for (int db_i = 0; db_i < db_locs.size(); db_i++) {
    v_fc[db_i] = 0;
    for (int defect_i = 0; defect_i < t_fc_locs.size(); defect_i++) {
      FPType db_x = db_locs[db_i].first;
      FPType db_y = db_locs[db_i].second;
      FPType db_z = 0;
      FPType defect_x = t_fc_locs[defect_i].x;
      FPType defect_y = t_fc_locs[defect_i].y;
      FPType defect_z = t_fc_locs[defect_i].z;
      FPType r = SimAnneal::distance(db_x, db_y, db_z, defect_x, defect_y, defect_z) * SimAnneal::db_distance_scale;
      v_fc[db_i] += SimAnneal::coulombicPotential(t_fcs[defect_i], 1,
        t_fc_eps_rs[defect_i], t_fc_lambdas[defect_i], r);
    }
  }
}

// SimAnneal (master) Implementation

SimAnneal::SimAnneal(SimParams &sparams)
{
  sim_params = sparams;
  initialize();
}


void SimAnneal::invokeSimAnneal()
{
  Logger log(saglobal::log_level);
  if (saglobal::log_level >= Logger::DBG) log.debug() << "Setting up SimAnnealThreads..." << std::endl;

  // Seeds belong to restart IDs, so worker scheduling never changes an RNG stream.
  std::vector<std::uint64_t> seeds(sim_params.num_instances);
  boost::random_device rd;
  for (int i=0; i<sim_params.num_instances; ++i) {
    seeds[i] = sim_params.deterministic_seed ? sim_params.random_seed + i
        : (static_cast<std::uint64_t>(rd()) << 32) | rd();
  }
  charge_results.assign(sim_params.num_instances, ThreadChargeResults());
  energy_results.assign(sim_params.num_instances, ThreadEnergyResults());
  suggested_gs_results.assign(sim_params.num_instances, ChargeConfigResult());
  anneal_threads.clear();
  std::atomic<int> next_restart(0);
  std::atomic<bool> failed(false);
  std::exception_ptr failure;
  std::mutex failure_mutex;
  auto worker = [&]() {
    try {
      while (!failed.load()) {
        const int id = next_restart.fetch_add(1);
        if (id >= sim_params.num_instances) break;
        SimAnnealThread annealer(id, seeds[id]);
        annealer.run();
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(failure_mutex);
      if (!failure) failure = std::current_exception();
      failed.store(true);
    }
  };
  try {
    for (int i=0; i<sim_params.num_workers; ++i)
      anneal_threads.emplace_back(worker);
  } catch (...) {
    failed.store(true);
    for (auto &th : anneal_threads) if (th.joinable()) th.join();
    anneal_threads.clear();
    throw;
  }
  for (auto &th : anneal_threads) th.join();
  anneal_threads.clear();
  if (failure) std::rethrow_exception(failure);

  if (saglobal::log_level >= Logger::DBG) log.debug() << "All simulations complete." << std::endl;
}

FPType SimAnneal::systemEnergy(const ublas::vector<int> &n_in, bool qubo)
{
  assert(n_in.size() > 0);

  //FPType E = 0.5 * ublas::inner_prod(n_in, ublas::prod(sim_params.v_ij, n_in))
    //- ublas::inner_prod(n_in, sim_params.v_ext);
  FPType E = ublas::inner_prod(n_in, sim_params.v_ext + sim_params.v_fc)
    + 0.5 * ublas::inner_prod(n_in, ublas::prod(sim_params.v_ij, n_in));
    

  if (qubo) {
    for (int n_i : n_in) {
      E += n_i * sim_params.mu;
    }
  }
  
  return E;
}

bool SimAnneal::isMetastable(const ublas::vector<int> &n_in)
{
  assert(n_in.size() > 0);
  Logger log(saglobal::log_level);

  const FPType &muzm = sparams->mu;
  const FPType &mupz = sparams->mu - constants::eta;
  const FPType &zero_equiv = constants::RECALC_STABILITY_ERR;

  ublas::vector<FPType> v_local(n_in.size());
  if (saglobal::log_level >= Logger::DBG) log.debug() << "V_i and Charge State Config " << n_in << ":" << std::endl;
  for (unsigned int i=0; i<n_in.size(); i++) {
    // calculate v_i
    v_local[i] = - (sim_params.v_ext[i] + sim_params.v_fc[i]);
    for (unsigned int j=0; j<n_in.size(); j++) {
      if (i == j) continue;
      v_local[i] -= sim_params.v_ij(i,j) * n_in[j];
    }
    if (saglobal::log_level >= Logger::DBG) log.debug() << "\tDB[" << i << "]: charge state=" << n_in[i]
      << ", v_local[i]=" << v_local[i] << " eV, and v_local[i]+muzm=" << v_local[i] + muzm << "eV" << std::endl;

    // return false if invalid
    if (!(   (n_in[i] == -1 && v_local[i] + muzm < zero_equiv)    // DB- valid condition
          || (n_in[i] == 1  && v_local[i] + mupz > - zero_equiv)  // DB+ valid condition
          || (n_in[i] == 0  && v_local[i] + muzm > - zero_equiv   // DB0 valid condition
                            && v_local[i] + mupz < zero_equiv))) {
      if (saglobal::log_level >= Logger::DBG) log.debug() << "config " << n_in << " has an invalid population, failed at index " << i << std::endl;
      if (saglobal::log_level >= Logger::DBG) log.debug() << "v_local[i]=" << v_local[i] << ", muzm=" << muzm << ", mupz=" << mupz << std::endl;
      return false;
    }
  }
  if (saglobal::log_level >= Logger::DBG) log.debug() << "config " << n_in << " has a valid population." << std::endl;

  auto hopDel = [v_local, n_in](const int &i, const int &j) -> FPType {
    int dn_i = (n_in[i]==-1) ? 1 : -1;
    int dn_j = - dn_i;
    return - v_local[i]*dn_i - v_local[j]*dn_j - sparams->v_ij(i,j);
  };

  for (unsigned int i=0; i<n_in.size(); i++) {
    // do nothing with DB+
    if (n_in[i] == 1)
      continue;

    for (unsigned int j=0; j<n_in.size(); j++) {
      // attempt hops from more negative charge states to more positive ones
      FPType E_del = hopDel(i, j);
      if ((n_in[j] > n_in[i]) && (E_del < -zero_equiv)) {
        if (saglobal::log_level >= Logger::DBG) log.debug() << "config " << n_in << " not stable since hopping from site "
          << i << " to " << j << " would result in an energy change of "
          << E_del << std::endl;
        return false;
      }
    }
  }
  if (saglobal::log_level >= Logger::DBG) log.debug() << "config " << n_in << " has a stable configuration." << std::endl;
  return true;
}

void SimAnneal::storeResults(SimAnnealThread *annealer, int thread_id)
{
  // Each restart owns a distinct slot; readers access results only after join.
  charge_results[thread_id].swap(annealer->db_charges);
  energy_results[thread_id].swap(annealer->config_energies);
  suggested_gs_results[thread_id] = std::move(annealer->suggestedConfig());

}

SuggestedResults SimAnneal::suggestedConfigResults(bool tidy)
{
  SuggestedResults filtered_results;
  if (tidy) {
    // deduplicate and recalculate energy
    std::unordered_set<std::string> config_set;
    //std::map< ublas::vector<int>, ChargeConfigResult > result_map;
    for (auto result : suggested_gs_results) {
      if (!result.initialized) {
        continue;
      }
      if (config_set.find(configToStr(result.config)) == config_set.end()) {
        config_set.insert(configToStr(result.config));
        if (isMetastable(result.config)) {
          result.system_energy = systemEnergy(result.config);
          filtered_results.push_back(result);
        }
      }
    }
  } else {
    // return every result that has been initialized
    for (auto result : suggested_gs_results)
      if (result.initialized)
        filtered_results.push_back(result);
  }
  return filtered_results;
}

FPType SimAnneal::coulombicPotential(FPType c_1, FPType c_2, FPType eps_r, FPType lambda, FPType r)
{
  return constants::Q0 / (4 * constants::PI * constants::EPS0 * eps_r) * exp(-r/(lambda*1e-9)) / r * c_1 * c_2;
}

FPType SimAnneal::distance(FPType x1, FPType y1, FPType z1, FPType x2, FPType y2, FPType z2)
{
  return sqrt(pow(x2-x1, 2) + pow(y2-y1, 2) + pow(z2-z1, 2));
}


// PRIVATE

void SimAnneal::initialize()
{
  Logger log(saglobal::log_level);
  SimParams &sp = sim_params;

#ifndef SIMANNEAL_HAVE_ACCELERATE
  if (sp.population_backend == PopulationBackend::Accelerate)
    throw std::invalid_argument("population_backend accelerate is unavailable in this build");
#endif
  if (sp.population_backend != PopulationBackend::Auto &&
      sp.population_backend != PopulationBackend::Portable &&
      sp.population_backend != PopulationBackend::Accelerate)
    throw std::invalid_argument("Unknown population_backend");

  if (sp.deterministic_seed && sp.random_seed > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("random_seed must be in [0,4294967295]");

  if (saglobal::log_level >= Logger::DBG) log.debug() << "Performing pre-calculations..." << std::endl;

  // set default values
  if (sp.v_freeze_init < 0)
    sp.v_freeze_init = fabs(sp.mu) / 2;
  if (sp.v_freeze_reset < 0)
    sp.v_freeze_reset = fabs(sp.mu);

  // apply schedule scaling
  sp.alpha = std::pow(std::exp(-1.), 1./(sp.T_e_inv_point * sp.anneal_cycles));
  sp.v_freeze_cycles = sp.v_freeze_end_point * sp.anneal_cycles;
  sp.v_freeze_step = sp.v_freeze_threshold / sp.v_freeze_cycles;

  if (saglobal::log_level >= Logger::DBG) log.debug() << "Anneal cycles: " << sp.anneal_cycles << ", alpha: "
    << sp.alpha << ", v_freeze_cycles: " << sp.v_freeze_cycles << std::endl;

  sp.result_queue_size = sp.anneal_cycles * sp.result_queue_factor;
  sp.result_queue_size = std::min(sp.result_queue_size, sp.anneal_cycles);
  sp.result_queue_size = std::max(sp.result_queue_size, 1);
  if (saglobal::log_level >= Logger::DBG) log.debug() << "Result queue size: " << sp.result_queue_size << std::endl;


  if (sp.preanneal_cycles > sp.anneal_cycles) {
    std::cerr << "Preanneal cycles > Anneal cycles";
    throw;
  }


  // phys
  sp.kT_min = constants::Kb * sp.T_min;
  sp.Kc = 1/(4 * constants::PI * sp.eps_r * constants::EPS0);

  // inter-db distances and voltages
  sp.population_finite_matrix = true;
  for (int i=0; i<sp.n_dbs; i++) {
    sp.db_r(i,i) = 0.;
    sp.v_ij(i,i) = 0.;
    for (int j=i+1; j<sp.n_dbs; j++) {
      sp.db_r(i,j) = db_distance_scale * distance(i,j);
      sp.v_ij(i,j) = interElecPotential(sp.db_r(i,j));
      if (!std::isfinite(sp.v_ij(i,j))) sp.population_finite_matrix = false;
      sp.db_r(j,i) = sp.db_r(i,j);
      sp.v_ij(j,i) = sp.v_ij(i,j);

      if (saglobal::log_level >= Logger::DBG) log.debug() << "db_r[" << i << "][" << j << "]=" << sp.db_r(i,j)
        << ", v_ij[" << i << "][" << j << "]=" << sp.v_ij(i,j) << std::endl;
    }
  }

  if (saglobal::log_level >= Logger::DBG) log.debug() << "Pre-calculations complete" << std::endl << std::endl;

  if (sp.hop_selection != UniformHop) {
    if (!std::isfinite(sp.hop_global_probability) || sp.hop_global_probability < 0
        || sp.hop_global_probability > 1)
      throw std::invalid_argument("hop_global_probability must be in [0,1]");
    if (sp.hop_selection == LocalRadiusHop)
      sp.hop_neighborhood.buildRadius(sp.db_r, sp.n_dbs, sp.hop_radius_nm);
    else
      sp.hop_neighborhood.build(sp.db_r, sp.n_dbs, sp.hop_neighbors,
          sp.hop_length_nm, sp.hop_selection == LocalDistanceHop);
  } else {
    sp.hop_neighborhood = HopNeighborhood();
  }

  // determine number of threads to run
  if (sp.num_instances == -1) {
    if (sp.n_dbs <= 9) {
      sp.num_instances = 16;
    } else if (sp.n_dbs <= 25) {
      sp.num_instances = 32;
    } else {
      sp.num_instances = 128;
    }
  }

  if (sp.num_instances <= 0) throw std::invalid_argument("num_instances must be positive or -1");
  if (sp.num_workers < 0) throw std::invalid_argument("num_workers must be nonnegative");
  if (sp.num_workers == 0)
    sp.num_workers = std::max(1u, std::thread::hardware_concurrency());
  sp.num_workers = std::min(sp.num_workers, sp.num_instances);
  charge_results.assign(sp.num_instances, ThreadChargeResults());
  energy_results.assign(sp.num_instances, ThreadEnergyResults());
  //cpu_times.resize(sp.num_instances);
  suggested_gs_results.assign(sp.num_instances, ChargeConfigResult());
}

FPType SimAnneal::distance(const int &i, const int &j)
{
  FPType x1 = sim_params.db_locs[i].first;
  FPType y1 = sim_params.db_locs[i].second;
  FPType x2 = sim_params.db_locs[j].first;
  FPType y2 = sim_params.db_locs[j].second;
  return sqrt(pow(x1-x2, 2.0) + pow(y1-y2, 2.0));
}

FPType SimAnneal::interElecPotential(const FPType &r)
{
  return constants::Q0 * sim_params.Kc * exp(-r/(sim_params.debye_length*1e-9)) / r;
}

FPType SimAnneal::hopEnergyDelta(ublas::vector<int> n_in, const int &from_ind, 
    const int &to_ind)
{
  // TODO make an efficient implementation with energy delta implementation
  FPType orig_energy = systemEnergy(n_in);
  int from_state = n_in[from_ind];
  n_in[from_ind] = n_in[to_ind];
  n_in[to_ind] = from_state;
  return systemEnergy(n_in) - orig_energy;
}



// SimAnnealThread Implementation

SimAnnealThread::SimAnnealThread(const int t_thread_id, const std::uint64_t seed)
  : thread_id(t_thread_id), gener(seed), dis01(0,1)
{
  muzm = sparams->mu;
  mupz = sparams->mu - constants::eta;
}

void SimAnnealThread::run()
{
  // initialize variables & perform pre-calculation
  kT = sparams->T_init*constants::Kb;
  v_freeze = 0.;
  t = 0;
  t_freeze = 0;
  t_phys_validity_check = 0;
  pop_schedule_phase = PopulationUpdateMode;

  // resize vectors
  n.resize(sparams->n_dbs);
  v_local.resize(sparams->n_dbs);
  db_charges.clear();
  config_energies.clear();
  db_charges.set_capacity(sparams->record_history ? sparams->result_queue_size : 0);
  config_energies.set_capacity(sparams->record_history ? sparams->result_queue_size : 0);
  suggested_gs = ChargeConfigResult();

  // SIM ANNEAL
  anneal();
}

void SimAnnealThread::anneal()
{
  typedef ublas::vector<int> OccListType;

  // Vars
  ublas::vector<int> dn(sparams->n_dbs);  // change of occupation for population update
  ublas::vector<FPType> population_v_delta(sparams->n_dbs);
#ifdef SIMANNEAL_HAVE_ACCELERATE
  const bool population_accelerate = sparams->population_backend != PopulationBackend::Portable;
  // Convert integer charge deltas once per changed population, reusing storage.
  std::vector<FPType> population_backend_input(population_accelerate ? sparams->n_dbs : 0);
#endif
  std::vector<unsigned> population_changed;
  population_changed.reserve(sparams->n_dbs);
  // Geometry initialization records this once for all independent restarts.
  const bool population_finite_matrix = sparams->population_finite_matrix;
  OccListType dbm_occ(sparams->n_dbs);                    // indices of DB- sites in n
  OccListType db0_occ(sparams->n_dbs);                    // indices of DB0 sites in n
  OccListType dbp_occ(sparams->n_dbs);                    // indices of DB+ sites in n
  const bool local_hops = sparams->hop_selection != UniformHop;
  const bool radius_hops = sparams->hop_selection == LocalRadiusHop;
  RadiusEligibleCache radius_cache;
  std::vector<int> neutral_slot(local_hops ? sparams->n_dbs : 0, -1);
  OccListType::iterator from_occ, to_occ;
  int dbm_occ_count=0, db0_occ_count=0, dbp_occ_count=0;
  int hop_attempts, max_hop_attempts;
  int from_ind, to_ind;               // hopping from n[from_ind] to n[to_ind]
  FPType hop_E_del;
  bool pop_changed;

  auto rand_charged_db_ind = [this, &dbm_occ, &dbp_occ, &dbm_occ_count,
                              &dbp_occ_count]
                                (OccListType::iterator &occ_it) mutable -> int
  {
    if (dbm_occ_count == 0 && dbp_occ_count == 0)
      return -1;
    int r_ind = randInt(0, dbm_occ_count + dbp_occ_count - 1);
    occ_it = (r_ind < dbm_occ_count) ? dbm_occ.begin() + r_ind
                                        : dbp_occ.begin() + (r_ind - dbm_occ_count);
    return *occ_it;
  };

  auto rand_neutral_db_ind = [this, &db0_occ, &db0_occ_count]
                                (OccListType::iterator &occ_it) mutable -> int
  {
    if (db0_occ_count == 0)
      return -1;
    int r_ind = randInt(0, db0_occ_count - 1);
    occ_it = db0_occ.begin() + r_ind;
    return *occ_it;
  };

  E_sys = systemEnergy();
  v_local = - (sparams->v_ext + sparams->v_fc) - ublas::prod(sparams->v_ij, n);

  Logger log(saglobal::log_level);

  // Run simulated annealing for predetermined time steps
  while(t < sparams->anneal_cycles) {
    //log.debug() << "Cycle " << t << ", kT=" << kT << ", v_freeze=" << v_freeze << std::endl;

    // Random population change, pop_changed is set to true if anything changes
    //log.debug() << "Before popgen: n=" << n << std::endl;
    genPopDelta(dn, pop_changed);
    if (pop_changed) {
      n += dn;
      if (radius_hops) population_changed.clear();
#ifdef SIMANNEAL_HAVE_ACCELERATE
      // Match the qualified dense backend: no unused sparse-index gathering.
      // Nonfinite geometries retain the portable dense behavior, including 0*NaN.
      if (population_accelerate && population_finite_matrix && !dn.empty()) {
        const std::size_t count = dn.size();
        for (std::size_t j = 0; j < count; ++j) {
          population_backend_input[j] = dn[j];
          if (radius_hops && dn[j] != 0) population_changed.push_back(static_cast<unsigned>(j));
        }
        FPType *population_output = &population_v_delta.data()[0];
        cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(count),
            static_cast<int>(count), 1.0, &sparams->v_ij.data()[0], static_cast<int>(count),
            population_backend_input.data(), 1, 0.0, population_output, 1);
        FPType population_linear_energy = 0;
        FPType population_quadratic_energy = 0;
        FPType *population_local = &v_local.data()[0];
        for (std::size_t i = 0; i < count; ++i) {
          population_linear_energy += population_local[i]*dn[i];
          population_quadratic_energy += dn[i]*population_output[i];
        }
        E_sys += -1 * population_linear_energy + 0.5 * population_quadratic_energy;
        for (std::size_t i = 0; i < count; ++i)
          population_local[i] -= population_output[i];
      } else
#endif
      {
        // Portable arithmetic and ascending-j sparse accumulation are unchanged.
        population_changed.clear();
        for (unsigned j = 0; j < dn.size(); ++j)
          if (dn[j] != 0) population_changed.push_back(j);
        if (population_finite_matrix && population_changed.size()*4 < dn.size()) {
          for (unsigned i = 0; i < dn.size(); ++i) {
            FPType sum = 0;
            for (const unsigned j : population_changed)
              sum += sparams->v_ij(i,j)*dn[j];
            population_v_delta[i] = sum;
          }
        } else {
          population_v_delta.assign(ublas::prod(sparams->v_ij, dn));
        }
        E_sys += -1 * ublas::inner_prod(v_local, dn)
          + 0.5 * ublas::inner_prod(dn, population_v_delta);
        v_local.minus_assign(population_v_delta);
      }

      // Occupation lists update
      int dbm_ind=0, db0_ind=0, dbp_ind=0;
      for (int db_ind=0; db_ind<sparams->n_dbs; db_ind++) {
        if (local_hops) neutral_slot[db_ind] = -1;
        if (n[db_ind]==-1) {
          dbm_occ[dbm_ind++] = db_ind;
        } else if (n[db_ind]==0) {
          if (local_hops) neutral_slot[db_ind] = db0_ind;
          db0_occ[db0_ind++] = db_ind;
        } else {
          dbp_occ[dbp_ind++] = db_ind;
        }
      }
      if (radius_hops && radius_cache.initialized)
        for (const unsigned site : population_changed)
          sparams->hop_neighborhood.setNeutral(site, n[site] == 0, radius_cache);
      dbm_occ_count = dbm_ind;
      db0_occ_count = db0_ind;
      dbp_occ_count = dbp_ind;
    }

    // Hopping - randomly hop electrons from higher occupancy sites to lower
    // occupancy sites
    hop_attempts = 0;
    max_hop_attempts = 0;
    if (dbm_occ_count + dbp_occ_count < sparams->n_dbs
        && db0_occ_count < sparams->n_dbs) {
      max_hop_attempts = std::max(dbm_occ_count+dbp_occ_count, db0_occ_count);
      max_hop_attempts *= sparams->hop_attempt_factor;
    }

    while (hop_attempts < max_hop_attempts) {
      from_ind = rand_charged_db_ind(from_occ);
      to_ind = -1;
      if (local_hops && (sparams->hop_global_probability == 0 ||
            (sparams->hop_global_probability < 1 &&
             dis01(gener) >= sparams->hop_global_probability))) {
        to_ind = radius_hops
            ? sparams->hop_neighborhood.selectRadius(from_ind, n, dis01(gener), radius_cache)
            : sparams->hop_neighborhood.select(from_ind, n, dis01(gener));
        if (to_ind == -2) {
          ++hop_attempts;
          continue;
        }
        if (to_ind >= 0) {
          assert(neutral_slot[to_ind] >= 0);
          to_occ = db0_occ.begin() + neutral_slot[to_ind];
        }
      }
      if (to_ind < 0) to_ind = rand_neutral_db_ind(to_occ);
      if (from_ind == -1 || to_ind == -1) {
        std::cerr << "Invalid hop index, this shouldn't happen." << std::endl;
        throw;
      }
      hop_E_del = hopEnergyDelta(from_ind, to_ind);
      if (acceptHop(hop_E_del)) {
        performHop(from_ind, to_ind, E_sys, hop_E_del);
        if (radius_hops && radius_cache.initialized) {
          sparams->hop_neighborhood.setNeutral(from_ind, n[from_ind] == 0, radius_cache);
          sparams->hop_neighborhood.setNeutral(to_ind, n[to_ind] == 0, radius_cache);
        }
        // update occupation indices list
        if (n[from_ind] - n[to_ind] < 2) {
          // hopping from DB- or DB+ to DB0
          if (local_hops) {
            neutral_slot[from_ind] = static_cast<int>(to_occ - db0_occ.begin());
            neutral_slot[to_ind] = -1;
          }
          int orig_from_ind = *from_occ;
          *from_occ = *to_occ;
          *to_occ = orig_from_ind;
        }
      }
      hop_attempts++;
    }

    // push back the new arrangement
    const bool pop_valid = populationValid();
    if (sparams->record_history) {
      db_charges.push_back(ChargeConfigResult(n, pop_valid, E_sys));
      config_energies.push_back(E_sys);
    }

    // keep track of suggested ground state
    if (pop_valid) {
      if (E_sys < suggested_gs.system_energy || suggested_gs.config.empty()) {
        suggested_gs.initialized = true;
        suggested_gs.config = n;
        suggested_gs.system_energy = E_sys;
        suggested_gs.pop_likely_stable = true;
      }
    }

    //log.debug() << "db_charges = " << n << std::endl;

    // perform time-step if not pre-annealing
    timeStep();
  }

  if (saglobal::log_level >= Logger::DBG) log.debug() << "Final db_charges = " << n
    << ", delta-based system energy = " << E_sys
    << ", recalculated system energy=" << systemEnergy() << std::endl;

  // Preserve an exportable final state even if no likely valid state was found.
  if (!suggested_gs.initialized)
    suggested_gs = ChargeConfigResult(n, populationValid(), E_sys);
  SimAnneal::storeResults(this, thread_id);
}

void SimAnnealThread::genPopDelta(ublas::vector<int> &dn, bool &changed)
{
  // DB- and DB+ sites can be flipped to DB0, DB0 sites can be flipped to either
  // DB- or DB+ depending on which one is enegetically closer.
  FPType prob;
  FPType x;
  int change_dir;
  changed = false;
  for (unsigned i=0; i<n.size(); i++) {
    if (n[i] == -1) {
      // Probability from DB- to DB0
      x = - (v_local[i] + muzm) + v_freeze;
      change_dir = 1;
    } else if (n[i] == 1) {
      // Probability from DB+ to DB0
      x = v_local[i] + mupz + v_freeze;
      change_dir = -1;
    } else {
      if (fabs(v_local[i] + muzm) < fabs(v_local[i] + mupz)) {
        // Closer to DB(0/-) transition level, probability from DB0 to DB-
        x = v_local[i] + muzm + v_freeze;
        change_dir = -1;
      } else {
        // Closer to DB(+/0) transition level, probability from DB0 to DB+
        x = - (v_local[i] + mupz) + v_freeze;
        change_dir = 1;
      }
    }
    prob = 1. / (1 + exp(x / kT));

    if (evalProb(prob)) {
      dn[i] = change_dir;
      changed = true;
    } else {
      dn[i] = 0;
    }
  }
}

void SimAnnealThread::performHop(const int &from_ind, const int &to_ind,
    FPType &E_sys, const FPType &E_del)
{
  int dn_i = (n[from_ind]==-1) ? 1 : -1;
  int dn_j = - dn_i;

  n[from_ind] += dn_i;
  n[to_ind] += dn_j;

  E_sys += E_del;
  // initialize() assigns each symmetric pair from the identical FP value.
  // Raw contiguous rows isolate matrix accessor overhead from the prior prototype.
  const std::size_t count = n.size();
  const FPType *row_i = &sparams->v_ij.data()[from_ind * count];
  const FPType *row_j = &sparams->v_ij.data()[to_ind * count];
  FPType *local = &v_local.data()[0];
  for (std::size_t k = 0; k < count; ++k)
    local[k] -= row_i[k]*dn_i + row_j[k]*dn_j;
}

void SimAnnealThread::timeStep()
{
  Logger log(saglobal::log_level);

  // always progress annealing schedule
  t++;

  // if preannealing, stop here
  if (t < sparams->preanneal_cycles)
    return;

  // decide what to do with v_freeze schedule
  switch (pop_schedule_phase) {
    case PopulationUpdateMode:
    {
      if (t_freeze < sparams->v_freeze_cycles) {
        t_freeze++;
      } else {
        if (!sparams->strategic_v_freeze_reset
            || (sparams->anneal_cycles - t) < sparams->v_freeze_cycles + sparams->phys_validity_check_cycles) {
          // if there aren't enough cycles left for another full v_freeze cycles, then
          // stop playing with t_freeze
          pop_schedule_phase = PopulationUpdateFinished;
        } else {
          // initiate physical validity check for the next phys_validity_check_cycles
          pop_schedule_phase = PhysicalValidityCheckMode;
          t_phys_validity_check = 0;
          phys_valid_count = 0;
          phys_invalid_count = 0;
        }
      }
      break;
    }
    case PhysicalValidityCheckMode:
    {
      if (t_phys_validity_check < sparams->phys_validity_check_cycles) {
        populationValid() ? phys_valid_count++ : phys_invalid_count++;
        t_phys_validity_check++;
      } else if (t_phys_validity_check >= sparams->phys_validity_check_cycles) {
        pop_schedule_phase = PopulationUpdateMode;
        t_freeze = 0;
        if (phys_valid_count < phys_invalid_count) {
          if (saglobal::log_level >= Logger::DBG) log.debug() << "Thread " << thread_id << ": t=" << t
            << ", charge config is " << n 
            << " which is physically invalid, resetting v_freeze." << std::endl;

          // reset v_freeze and temperature
          v_freeze = sparams->v_freeze_reset;
          if (sparams->reset_T_during_v_freeze_reset)
            kT = sparams->T_init*constants::Kb;
        }
      }
      break;
    }
    case PopulationUpdateFinished:
      break;
    default:
      std::cerr << "Invalid PopulationSchedulePhase.";
      throw;
  }

  // update parameters according to schedule
  if (sparams->T_schedule == ExponentialSchedule)
    kT = sparams->kT_min + (kT - sparams->kT_min) * sparams->alpha;
  else if (sparams->T_schedule == LinearSchedule)
    kT = std::max(sparams->kT_min, kT - sparams->alpha);

  // update v_freeze
  if (v_freeze < sparams->v_freeze_threshold)
    v_freeze += sparams->v_freeze_step;
}

bool SimAnnealThread::acceptHop(const FPType &v_diff)
{
  if (v_diff < 0)
    return true;

  // some acceptance function, acceptance probability falls off exponentially
  FPType prob = exp(-v_diff/kT);

  return evalProb(prob);
}

bool SimAnnealThread::evalProb(const FPType &prob)
{
  return prob >= dis01(gener);
}

int SimAnnealThread::randInt(const int &min, const int &max)
{
  RandIntDist dis(min,max);
  return dis(gener);
}

FPType SimAnnealThread::systemEnergy() const
{
  return ublas::inner_prod(n, (sparams->v_ext + sparams->v_fc))
    + 0.5 * ublas::inner_prod(n, ublas::prod(sparams->v_ij, n));
}

/*
FPType SimAnnealThread::totalCoulombPotential(ublas::vector<int> &config) const
{
  return 0.5 * ublas::inner_prod(config, ublas::prod(sparams->v_ij, config));
}
*/

FPType SimAnnealThread::hopEnergyDelta(const int &i, const int &j)
{
  //return v_local[i] - v_local[j] - sparams->v_ij(i,j);
  int dn_i = (n[i]==-1) ? 1 : -1;
  int dn_j = - dn_i;
  return - v_local[i]*dn_i - v_local[j]*dn_j - sparams->v_ij(i,j);
}

bool SimAnnealThread::populationValid() const
{
  // Check whether v_local at each site meets population validity constraints
  // Note that v_local components have flipped signs from E_sys
  bool valid;
  const FPType &zero_equiv = constants::POP_STABILITY_ERR;
  for (int i=0; i<sparams->n_dbs; i++) {
    valid = ((n[i] == -1 && v_local[i] + muzm < zero_equiv)   // DB- condition
          || (n[i] == 1  && v_local[i] + mupz > -zero_equiv)  // DB+ condition
          || (n[i] == 0  && v_local[i] + muzm > -zero_equiv
                         && v_local[i] + mupz < zero_equiv));
    if (!valid) {
      return false;
    }
  }
  return true;
}
