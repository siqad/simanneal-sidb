// @file:     sim_anneal.cc
// @author:   Samuel
// @created:  2017.08.23
// @editted:  2019.02.01
// @license:  Apache License 2.0
//
// @desc:     Simulated annealing physics engine

#include "simanneal.h"
#include <ctime>
#include <algorithm>
#include <unordered_set>

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

void SimParams::setDBLocs(const std::vector<LatCoord> &t_db_locs)
{
  std::vector<EuclCoord> db_locs;
  for (LatCoord lat_coord : t_db_locs) {
    assert(lat_coord.size() == 3);
    db_locs.push_back(latToEuclCoord(lat_coord[0], lat_coord[1], lat_coord[2]));
  }
  setDBLocs(db_locs);
}

EuclCoord SimParams::latToEuclCoord(const int &n, const int &m, const int &l)
{
  FPType x = n * constants::lat_a;
  FPType y = m * constants::lat_b + l * constants::lat_c;
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
  log.debug() << "Setting up SimAnnealThreads..." << std::endl;

  // spawn all the threads
  for (int i=0; i<sim_params.num_instances; i++) {
    boost::random_device rd;
    std::uint64_t seed = rd();
    seed = (seed << 32) | rd();
    SimAnnealThread annealer(i, seed);
    std::thread th(&SimAnnealThread::run, annealer);
    anneal_threads.push_back(std::move(th));
  }

  log.debug() << "Wait for simulations to complete." << std::endl;

  // wait for threads to complete
  for (auto &th : anneal_threads) {
    th.join();
  }

  log.debug() << "All simulations complete." << std::endl;
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
  log.debug() << "V_i and Charge State Config " << n_in << ":" << std::endl;
  for (unsigned int i=0; i<n_in.size(); i++) {
    // calculate v_i
    v_local[i] = - (sim_params.v_ext[i] + sim_params.v_fc[i]);
    for (unsigned int j=0; j<n_in.size(); j++) {
      if (i == j) continue;
      v_local[i] -= sim_params.v_ij(i,j) * n_in[j];
    }
    log.debug() << "\tDB[" << i << "]: charge state=" << n_in[i]
      << ", v_local[i]=" << v_local[i] << " eV, and v_local[i]+muzm=" << v_local[i] + muzm << "eV" << std::endl;

    // return false if invalid
    if (!(   (n_in[i] == -1 && v_local[i] + muzm < zero_equiv)    // DB- valid condition
          || (n_in[i] == 1  && v_local[i] + mupz > - zero_equiv)  // DB+ valid condition
          || (n_in[i] == 0  && v_local[i] + muzm > - zero_equiv   // DB0 valid condition
                            && v_local[i] + mupz < zero_equiv))) {
      log.debug() << "config " << n_in << " has an invalid population, failed at index " << i << std::endl;
      log.debug() << "v_local[i]=" << v_local[i] << ", muzm=" << muzm << ", mupz=" << mupz << std::endl;
      return false;
    }
  }
  log.debug() << "config " << n_in << " has a valid population." << std::endl;

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
        log.debug() << "config " << n_in << " not stable since hopping from site "
          << i << " to " << j << " would result in an energy change of "
          << E_del << std::endl;
        return false;
      }
    }
  }
  log.debug() << "config " << n_in << " has a stable configuration." << std::endl;
  return true;
}

void SimAnneal::storeResults(SimAnnealThread *annealer, int thread_id)
{
  result_store_mutex.lock();

  charge_results[thread_id] = annealer->db_charges;
  energy_results[thread_id] = annealer->config_energies;
  //cpu_times[thread_id] = annealer->CPUTime();

  suggested_gs_results[thread_id] = annealer->suggestedConfig();

  result_store_mutex.unlock();
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

  log.debug() << "Performing pre-calculations..." << std::endl;

  // set default values
  if (sp.v_freeze_init < 0)
    sp.v_freeze_init = fabs(sp.mu) / 2;
  if (sp.v_freeze_reset < 0)
    sp.v_freeze_reset = fabs(sp.mu);

  // apply schedule scaling
  sp.alpha = std::pow(std::exp(-1.), 1./(sp.T_e_inv_point * sp.anneal_cycles));
  sp.v_freeze_cycles = sp.v_freeze_end_point * sp.anneal_cycles;
  sp.v_freeze_step = sp.v_freeze_threshold / sp.v_freeze_cycles;

  log.debug() << "Anneal cycles: " << sp.anneal_cycles << ", alpha: " 
    << sp.alpha << ", v_freeze_cycles: " << sp.v_freeze_cycles << std::endl;

  sp.result_queue_size = sp.anneal_cycles * sp.result_queue_factor;
  sp.result_queue_size = std::min(sp.result_queue_size, sp.anneal_cycles);
  sp.result_queue_size = std::max(sp.result_queue_size, 1);
  log.debug() << "Result queue size: " << sp.result_queue_size << std::endl;


  if (sp.preanneal_cycles > sp.anneal_cycles) {
    std::cerr << "Preanneal cycles > Anneal cycles";
    throw;
  }


  // phys
  sp.kT_min = constants::Kb * sp.T_min;
  sp.Kc = 1/(4 * constants::PI * sp.eps_r * constants::EPS0);

  // inter-db distances and voltages
  for (int i=0; i<sp.n_dbs; i++) {
    sp.db_r(i,i) = 0.;
    sp.v_ij(i,i) = 0.;
    for (int j=i+1; j<sp.n_dbs; j++) {
      sp.db_r(i,j) = db_distance_scale * distance(i,j);
      sp.v_ij(i,j) = interElecPotential(sp.db_r(i,j));
      sp.db_r(j,i) = sp.db_r(i,j);
      sp.v_ij(j,i) = sp.v_ij(i,j);

      log.debug() << "db_r[" << i << "][" << j << "]=" << sp.db_r(i,j) 
        << ", v_ij[" << i << "][" << j << "]=" << sp.v_ij(i,j) << std::endl;
    }
  }

  log.debug() << "Pre-calculations complete" << std::endl << std::endl;

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

  charge_results.resize(sp.num_instances);
  energy_results.resize(sp.num_instances);
  //cpu_times.resize(sp.num_instances);
  suggested_gs_results.resize(sp.num_instances);
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
  db_charges.resize(sparams->result_queue_size);
  config_energies.resize(sparams->result_queue_size);

  // SIM ANNEAL
  anneal();
}

void SimAnnealThread::anneal()
{
  typedef ublas::vector<int> OccListType;

  // Vars
  ublas::vector<int> dn(sparams->n_dbs);  // change of occupation for population update
  OccListType dbm_occ(sparams->n_dbs);                    // indices of DB- sites in n
  OccListType db0_occ(sparams->n_dbs);                    // indices of DB0 sites in n
  OccListType dbp_occ(sparams->n_dbs);                    // indices of DB+ sites in n
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
      E_sys += -1 * ublas::inner_prod(v_local, dn)
        + 0.5 * ublas::inner_prod(dn, ublas::prod(sparams->v_ij, dn));
      v_local -= ublas::prod(sparams->v_ij, dn);

      // Occupation lists update
      int dbm_ind=0, db0_ind=0, dbp_ind=0;
      for (int db_ind=0; db_ind<sparams->n_dbs; db_ind++) {
        if (n[db_ind]==-1) {
          dbm_occ[dbm_ind++] = db_ind;
        } else if (n[db_ind]==0) {
          db0_occ[db0_ind++] = db_ind;
        } else {
          dbp_occ[dbp_ind++] = db_ind;
        }
        dbm_occ_count = dbm_ind;
        db0_occ_count = db0_ind;
        dbp_occ_count = dbp_ind;
      }
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
      to_ind = rand_neutral_db_ind(to_occ);
      if (from_ind == -1 || to_ind == -1) {
        std::cerr << "Invalid hop index, this shouldn't happen." << std::endl;
        throw;
      }
      hop_E_del = hopEnergyDelta(from_ind, to_ind);
      if (acceptHop(hop_E_del)) {
        performHop(from_ind, to_ind, E_sys, hop_E_del);
        // update occupation indices list
        if (n[from_ind] - n[to_ind] < 2) {
          // hopping from DB- or DB+ to DB0
          int orig_from_ind = *from_occ;
          *from_occ = *to_occ;
          *to_occ = orig_from_ind;
        }
      }
      hop_attempts++;
    }

    // push back the new arrangement
    db_charges.push_back(ChargeConfigResult(n, 
          populationValid(), E_sys));
    config_energies.push_back(E_sys);

    // keep track of suggested ground state
    if (populationValid()) {
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

  log.debug() << "Final db_charges = " << n
    << ", delta-based system energy = " << E_sys
    << ", recalculated system energy=" << systemEnergy() << std::endl;

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
  ublas::matrix_column<ublas::matrix<FPType>> v_i (sparams->v_ij, from_ind);
  ublas::matrix_column<ublas::matrix<FPType>> v_j (sparams->v_ij, to_ind);
  v_local -= v_i*dn_i + v_j*dn_j;
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
          log.debug() << "Thread " << thread_id << ": t=" << t 
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
