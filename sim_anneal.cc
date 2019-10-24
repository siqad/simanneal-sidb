// @file:     sim_anneal.cc
// @author:   Samuel
// @created:  2017.08.23
// @editted:  2019.02.01
// @license:  Apache License 2.0
//
// @desc:     Simulated annealing physics engine

#include "sim_anneal.h"
#include <ctime>
#include <algorithm>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

// thread CPU time for Linux
#include <pthread.h>
#include <time.h>

//#define STEADY_THREASHOLD 700       //arbitrary value used in restarting

using namespace phys;

// static variables
SimParams SimAnneal::sim_params;
boost::mutex SimAnneal::result_store_mutex;
FPType SimAnneal::db_distance_scale = 1E-10;
AllChargeResults SimAnneal::charge_results;
AllEnergyResults SimAnneal::energy_results;
AllCPUTimes SimAnneal::cpu_times;
AllSuggestedConfigResults SimAnneal::suggested_config_results;

// alias for the commonly used sim_params static variable
constexpr auto sparams = &SimAnneal::sim_params;


// SimAnneal (master) Implementation

SimAnneal::SimAnneal()
{
  initialize();
}

void SimAnneal::initialize()
{
  Logger log(global::log_level);
  log.echo() << "Performing pre-calculations..." << std::endl;

  if (sparams->preanneal_cycles > sparams->anneal_cycles) {
    std::cerr << "Preanneal cycles > Anneal cycles";
    throw;
  }

  // phys
  sim_params.kT_min = constants::Kb * sim_params.T_min;
  sim_params.Kc = 1/(4 * constants::PI * sim_params.epsilon_r * constants::EPS0);

  // DB distance and potentials
  sim_params.db_r.resize(sim_params.n_dbs, sim_params.n_dbs);
  sim_params.v_ext.resize(sim_params.n_dbs);
  sim_params.v_ij.resize(sim_params.n_dbs, sim_params.n_dbs);

  // inter-db distances and voltages
  for (int i=0; i<sim_params.n_dbs; i++) {
    sim_params.db_r(i,i) = 0.;
    sim_params.v_ij(i,i) = 0.;
    for (int j=i+1; j<sim_params.n_dbs; j++) {
      sim_params.db_r(i,j) = db_distance_scale * distance(i,j);
      sim_params.v_ij(i,j) = interElecPotential(sim_params.db_r(i,j));
      sim_params.db_r(j,i) = sim_params.db_r(i,j);
      sim_params.v_ij(j,i) = sim_params.v_ij(i,j);

      log.debug() << "db_r[" << i << "][" << j << "]=" << sim_params.db_r(i,j) 
        << ", v_ij[" << i << "][" << j << "]=" << sim_params.v_ij(i,j) << std::endl;
    }

    // TODO add electrode effect to v_ext

    //sim_accessor.v_ext[i] = sim_accessor.mu;
    sim_params.v_ext[i] = 0;
  }

  log.echo() << "Pre-calculations complete" << std::endl << std::endl;

  // determine number of threads to run
  if (sim_params.num_threads == -1) {
    if (sim_params.n_dbs <= 9) {
      sim_params.num_threads = 8;
    } else if (sim_params.n_dbs <= 25) {
      sim_params.num_threads = 16;
    } else {
      sim_params.num_threads = 128;
    }
  }

  charge_results.resize(sim_params.num_threads);
  energy_results.resize(sim_params.num_threads);
  cpu_times.resize(sim_params.num_threads);
  suggested_config_results.resize(sim_params.num_threads);
}

void SimAnneal::invokeSimAnneal()
{
  Logger log(global::log_level);
  log.echo() << "Setting up SimAnnealThreads..." << std::endl;

  // spawn all the threads
  for (int i=0; i<sim_params.num_threads; i++) {
    SimAnnealThread annealer(i);
    boost::thread th(&SimAnnealThread::run, annealer);
    anneal_threads.push_back(std::move(th));
  }

  log.echo() << "Wait for simulations to complete." << std::endl;

  // wait for threads to complete
  for (auto &th : anneal_threads) {
    th.join();
  }

  log.echo() << "All simulations complete." << std::endl;
}

FPType SimAnneal::systemEnergy(const ublas::vector<int> &n_in, bool qubo)
{
  assert(n_in.size() > 0);

  float E = 0.5 * ublas::inner_prod(n_in, ublas::prod(sim_params.v_ij, n_in))
    - ublas::inner_prod(n_in, sim_params.v_ext);

  if (qubo) {
    for (int n_i : n_in) {
      E -= n_i * sim_params.mu;
    }
  }
  
  return E;
}

bool SimAnneal::populationValidity(const ublas::vector<int> &n_in)
{
  assert(n_in.size() > 0);
  Logger log(global::log_level);

  const FPType &mu = sparams->mu;
  const FPType &eta = constants::eta;
  const FPType &zero_equiv = constants::RECALC_STABILITY_ERR;
  FPType v_i;
  log.debug() << "V_i and Charge State (Opposite sign from paper, here +1 = DB-) Config " << n_in << ":" << std::endl;
  for (unsigned int i=0; i<n_in.size(); i++) {
    // calculate v_i
    v_i = sim_params.v_ext[i];
    for (unsigned int j=0; j<n_in.size(); j++) {
      if (i == j) continue;
      v_i -= sim_params.v_ij(i,j) * n_in[j];
    }
    log.debug() << "\tDB[" << i << "]: charge state=" << n_in[i]
      << " and V_i=" << v_i << " eV" << std::endl;

    // return false if invalid
    if (!(   (n_in[i] == 1  && v_i + mu >= -zero_equiv)       // DB- valid condition
          || (n_in[i] == -1 && v_i + mu + eta <= zero_equiv)  // DB+ valid condition
          || (n_in[i] == 0  && v_i + mu < zero_equiv          // DB0 valid condition
                            && v_i + mu + eta > -zero_equiv))) {
      log.debug() << "config " << n_in << " has an invalid population, failed at index " << i << std::endl;
      log.debug() << "v_i=" << v_i << ", mu=" << mu << ", eta=" << eta << std::endl;
      return false;
    }
  }
  log.debug() << "config " << n_in << " has a valid population." << std::endl;
  return true;
}

bool SimAnneal::locallyMinimal(const ublas::vector<int> &n_in)
{
  assert(n_in.size() > 0);
  Logger log(global::log_level);

  for (unsigned int i=0; i<n_in.size(); i++) {
    // nothing to do with DB+ states
    if (n_in[i] == -1)
      continue;

    const FPType &zero_equiv = constants::RECALC_STABILITY_ERR;
    for (unsigned int j=0; j<n_in.size(); j++) {
      // only higher occupancy states can hop to lower ones:
      if (n_in[i] > n_in[j]) {
        FPType E_del = hopEnergyDelta(n_in, i, j);
        if (E_del < -zero_equiv) {
          log.debug() << "config " << n_in << " not stable since hopping from site "
            << i << " to " << j << " would result in an energy reduction of "
            << E_del << std::endl;
          return false;
        }
      }
    }
  }
  log.debug() << "config " << n_in << " has a stable configuration." << std::endl;
  return true;
}

void SimAnneal::storeResults(SimAnnealThread *annealer, int thread_id){
  result_store_mutex.lock();

  charge_results[thread_id] = annealer->db_charges;
  energy_results[thread_id] = annealer->config_energies;
  cpu_times[thread_id] = annealer->CPUTime();
  suggested_config_results[thread_id] = annealer->suggestedConfig();

  result_store_mutex.unlock();
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
  return constants::Q0 * sim_params.Kc * exp(-r/sim_params.debye_length) / r;
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

SimAnnealThread::SimAnnealThread(const int t_thread_id)
  : thread_id(t_thread_id)
{
  // initialize rng
  rng.seed(std::time(NULL)*thread_id+4065);
  dis01 = boost::random::uniform_real_distribution<FPType>(0,1);
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
  v_local.resize(sparams->n_dbs);

  db_charges.resize(sparams->result_queue_size);
  n.resize(sparams->n_dbs);

  config_energies.resize(sparams->result_queue_size);

  // SIM ANNEAL
  anneal();
}

void SimAnnealThread::anneal()
{
  typedef boost::numeric::ublas::vector<int> OccListType;

  // Vars
  boost::numeric::ublas::vector<int> dn(sparams->n_dbs);  // change of occupation for population update
  OccListType doubly_occ(sparams->n_dbs);            // indices of DB- sites in n
  OccListType singly_occ(sparams->n_dbs);            // indices of DB0 sites in n
  OccListType no_occ(sparams->n_dbs);                // indices of DB+ sites in n
  OccListType::iterator from_occ, to_occ;
  int doubly_occ_count=0, singly_occ_count=0, no_occ_count=0;
  int hop_attempts, max_hop_attempts;
  int from_ind, to_ind;         // hopping from n[from_ind] to n[to_ind]
  FPType hop_E_del;
  bool pop_changed;

  // Randomly select an occupied (DB- or DB0) site. If there aren't any DB+ sites,
  // then don't bother selected DB0 sites to hop from.
  auto rand_occ_db_ind = [this, &doubly_occ, &singly_occ, &doubly_occ_count, 
                          &singly_occ_count, &no_occ_count]
                            (OccListType::iterator &occ_it) mutable -> int
  {
    if (doubly_occ_count == 0 && singly_occ_count == 0)
      return -1;
    int max_ind = (no_occ_count > 0) ? doubly_occ_count + singly_occ_count
                                     : doubly_occ_count;
    int r_ind = randInt(0, max_ind - 1);
    occ_it = (r_ind < doubly_occ_count) ? doubly_occ.begin() + r_ind
                                        : singly_occ.begin() + r_ind - doubly_occ_count;
    return *occ_it;
  };

  // Randomly select a site with vacancy (DB0 or DB+). If the from_ind site 
  // is DB0, then the target site must be DB+.
  auto rand_vac_db_ind = [this, &singly_occ, &no_occ, &singly_occ_count, 
                          &no_occ_count, &from_occ](OccListType::iterator &occ_it) mutable -> int
  {
    if (singly_occ_count == 0 && no_occ_count == 0)
      return -1;
    int min_ind = (n[*from_occ]==0) ? singly_occ_count : 0; // omit DB0 sites from selection if hopping from DB0 site
    int max_ind = singly_occ_count + no_occ_count;
    int r_ind = randInt(min_ind, max_ind - 1);
    occ_it = (r_ind < singly_occ_count) ? singly_occ.begin() + r_ind
                                        : no_occ.begin() + r_ind - singly_occ_count;
    return *occ_it;
  };

  E_sys = systemEnergy();
  v_local = sparams->v_ext - ublas::prod(sparams->v_ij, n);

  Logger log(global::log_level);

  // Run simulated annealing for predetermined time steps
  while(t < sparams->anneal_cycles) {
    //log.debug() << "Cycle " << t << ", kT=" << kT << ", v_freeze=" << v_freeze << std::endl;

    // Random population change, pop_changed is set to true if anything changes
    genPopDelta(dn, pop_changed);
    if (pop_changed) {
      n += dn;
      E_sys += -1 * ublas::inner_prod(v_local, dn) + totalCoulombPotential(dn);
      v_local -= ublas::prod(sparams->v_ij, dn);

      // Occupation lists update
      int d_ind=0, s_ind=0, n_ind=0;
      for (int db_ind=0; db_ind<sparams->n_dbs; db_ind++) {
        if (n[db_ind]==1) {
          doubly_occ[d_ind++] = db_ind;
        } else if (n[db_ind]==0) {
          singly_occ[s_ind++] = db_ind;
        } else {
          no_occ[n_ind++] = db_ind;
        }
        doubly_occ_count = d_ind;
        singly_occ_count = s_ind;
        no_occ_count = n_ind;
      }
    }

    // Hopping - randomly hop electrons from higher occupancy sites to lower
    // occupancy sites
    hop_attempts = 0;
    if (doubly_occ_count != sparams->n_dbs && singly_occ_count != sparams->n_dbs
        && no_occ_count != sparams->n_dbs) {
      max_hop_attempts = doubly_occ_count + singly_occ_count;
      max_hop_attempts *= sparams->hop_attempt_factor;
      while (hop_attempts < max_hop_attempts) {
        from_ind = rand_occ_db_ind(from_occ);
        to_ind = rand_vac_db_ind(to_occ);
        if (from_ind == -1 || to_ind == -1) {
          std::cerr << "Cycle: " << t << ", attempting hop from ind " 
            << from_ind << " to " << to_ind << " when the configuration is "
            << n << " which is an invalid operation." << std::endl;
          throw;
        }
        hop_E_del = hopEnergyDelta(from_ind, to_ind);
        if (acceptHop(hop_E_del)) {
          performHop(from_ind, to_ind, E_sys, hop_E_del);
          // update occupation indices list
          if (n[from_ind]-n[to_ind] < 2) {
            // hopping from DB- to DB0 or DB0 to DB+
            int orig_from_ind = *from_occ;
            *from_occ = *to_occ;
            *to_occ = orig_from_ind;
          } else {
            // hopping from DB- to DB+, both becomes DB0
            singly_occ[singly_occ_count++] = from_ind;
            singly_occ[singly_occ_count++] = to_ind;
            *from_occ = doubly_occ[--doubly_occ_count];
            *to_occ = no_occ[--no_occ_count];
          }
        }
        hop_attempts++;
      }
    }

    // push back the new arrangement
    db_charges.push_back(ElecChargeConfigResult(n, 
          populationValid(), E_sys));
    config_energies.push_back(E_sys);

    // keep track of suggested ground state
    if (populationValid()) {
      if (E_sys_valid_gs == 0 || E_sys < E_sys_valid_gs) {
        E_sys_valid_gs = E_sys;
        n_valid_gs = n;
      }
    } else if ((sparams->anneal_cycles - t) <= sparams->result_queue_size) {
      if (E_sys_invalid_gs == 0 || E_sys < E_sys_invalid_gs) {
        E_sys_invalid_gs = E_sys;
        n_invalid_gs = n;
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
    if (n[i] == 1) {
      // Probability from DB- to DB0, in paper P_{1->0}
      x = v_local[i] + sparams->mu + v_freeze;
      change_dir = -1;
    } else if (n[i] == -1) {
      // Probability from DB+ to DB0, in paper P_{-1->0}
      x = -(v_local[i] + sparams->mu + constants::eta) + v_freeze;
      change_dir = 1;
    } else {
      if (abs(v_local[i] - sparams->mu) > abs(v_local[i] - (sparams->mu + constants::eta))) {
        // Closer to DB(+/0) transition level, probability from DB0 to DB+, in paper P_{0->-1}
        x = v_local[i] + sparams->mu + constants::eta + v_freeze;
        change_dir = -1;
      } else {
        // Closer to DB(0/-) transition level, probability from DB0 to DB-, in paper P_{0->1}
        x = -(v_local[i] + sparams->mu) + v_freeze;
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
    float &E_sys, const float &E_del)
{
  n[from_ind]--;
  n[to_ind]++;

  E_sys += E_del;
  ublas::matrix_column<ublas::matrix<FPType>> v_i (sparams->v_ij, from_ind);
  ublas::matrix_column<ublas::matrix<FPType>> v_j (sparams->v_ij, to_ind);
  v_local += v_i - v_j;
}

void SimAnnealThread::timeStep()
{
  Logger log(global::log_level);

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
  boost::variate_generator<boost::random::mt19937&, boost::random::uniform_real_distribution<FPType>> rnd_gen(rng, dis01);
  return prob >= rnd_gen();
}

int SimAnnealThread::randInt(const int &min, const int &max)
{
  boost::random::uniform_int_distribution<int> dis(min,max);
  return dis(rng);
}

FPType SimAnnealThread::systemEnergy() const
{
  assert(sparams->n_dbs > 0);
  return 0.5 * ublas::inner_prod(n, ublas::prod(sparams->v_ij, n))
    - ublas::inner_prod(n, sparams->v_ext);
}

FPType SimAnnealThread::totalCoulombPotential(ublas::vector<int> &config) const
{
  return 0.5 * ublas::inner_prod(config, ublas::prod(sparams->v_ij, config));
}

FPType SimAnnealThread::hopEnergyDelta(const int &i, const int &j)
{
  return v_local[i] - v_local[j] - sparams->v_ij(i,j);
}

bool SimAnnealThread::populationValid() const
{
  // Check whether v_local at each site meets population validity constraints
  // Note that v_local components have flipped signs from E_sys
  bool valid;
  const FPType &mu = sparams->mu;
  const FPType &eta = constants::eta;
  const FPType &zero_equiv = constants::POP_STABILITY_ERR;
  for (int i=0; i<sparams->n_dbs; i++) {
    if (   (n[i] == 1  && v_local[i] + mu >= -zero_equiv)       // DB- condition
        || (n[i] == -1 && v_local[i] + mu + eta < zero_equiv)   // DB+ condition
        || (n[i] == 0  && v_local[i] + mu < zero_equiv          // DB0 condition
                       && v_local[i] + mu + eta > -zero_equiv)) {
      valid = true;
    } else {
      valid = false;
    }

    if (!valid) {
      return false;
    }
  }
  return true;
}
