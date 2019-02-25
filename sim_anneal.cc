// @file:     sim_anneal.cc
// @author:   Samuel
// @created:  2017.08.23
// @editted:  2019.02.01
// @license:  Apache License 2.0
//
// @desc:     Simulated annealing physics engine

#include "sim_anneal.h"
#include <ctime>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

//#define STEADY_THREASHOLD 700       //arbitrary value used in restarting

using namespace phys;

// static variables
SimParams SimAnneal::sim_params;
boost::mutex SimAnneal::result_store_mutex;
float SimAnneal::db_distance_scale = 1E-10;
AllChargeResults SimAnneal::charge_results;
AllEnergyResults SimAnneal::energy_results;

// alias for the commonly used sim_params static variable
constexpr auto sparams = &SimAnneal::sim_params;


// SimAnneal (master) Implementation

SimAnneal::SimAnneal()
{
  initialize();
}

void SimAnneal::initialize()
{
  std::cout << "Performing pre-calculations..." << std::endl;

  if (sparams->preanneal_cycles > sparams->anneal_cycles)
    throw "Preanneal cycles > Anneal cycles";

  // phys
  sim_params.kT_min = constants::Kb * sim_params.T_min;
  sim_params.Kc = 1/(4 * constants::PI * sim_params.epsilon_r * constants::EPS0);

  // DB distance and potentials
  sim_params.db_r.resize(sim_params.n_dbs, sim_params.n_dbs);
  sim_params.v_ext.resize(sim_params.n_dbs);
  sim_params.v_ij.resize(sim_params.n_dbs, sim_params.n_dbs);

  // inter-db distances and voltages
  for (int i=0; i<sim_params.n_dbs; i++) {
    sim_params.db_r(i,i) = 0;
    sim_params.v_ij(i,i) = 0;
    for (int j=i+1; j<sim_params.n_dbs; j++) {
      sim_params.db_r(i,j) = db_distance_scale * distance(i,j);
      sim_params.v_ij(i,j) = interElecPotential(sim_params.db_r(i,j));
      sim_params.db_r(j,i) = sim_params.db_r(i,j);
      sim_params.v_ij(j,i) = sim_params.v_ij(i,j);

      std::cout << "db_r[" << i << "][" << j << "]=" << sim_params.db_r(i,j) 
        << ", v_ij[" << i << "][" << j << "]=" << sim_params.v_ij(i,j) << std::endl;
    }

    // TODO add electrode effect to v_ext

    //sim_accessor.v_ext[i] = sim_accessor.mu;
    sim_params.v_ext[i] = 0;
  }

  std::cout << "Pre-calculations complete" << std::endl << std::endl;

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
  //numElecStore.resize(sim_accessor.num_threads);
}

void SimAnneal::invokeSimAnneal()
{
  std::cout << "Setting up SimAnnealThreads..." << std::endl;

  // spawn all the threads
  for (int i=0; i<sim_params.num_threads; i++) {
    SimAnnealThread annealer(i);
    boost::thread th(&SimAnnealThread::run, annealer);
    anneal_threads.push_back(std::move(th));
  }

  std::cout << "Wait for simulations to complete." << std::endl;

  // wait for threads to complete
  for (auto &th : anneal_threads) {
    th.join();
  }

  std::cout << "All simulations complete." << std::endl;
}

float SimAnneal::systemEnergy(std::string n_in, int n_dbs)
{
  assert(n_dbs > 0);
  assert(n_in.length() == n_dbs);
  // convert string of 0 and 1 to ublas vector
  ublas::vector<int> n_int(n_in.length());
  for (int i=0; i<n_dbs; i++) {
    // ASCII char to int with the correct integer
    n_int[i] = n_in.at(i) - '0';
  }

  float v = 0;
  for(int i=0; i<n_dbs; i++) {
    //v -= mu + v_ext[i] * n[i];
    v -= sim_params.v_ext[i] * n_int[i];
    for(int j=i+1; j<n_dbs; j++)
      v += sim_params.v_ij(i,j) * n_int[i] * n_int[j];
  }
  return v;
}

void SimAnneal::storeResults(SimAnnealThread *annealer, int thread_id){
  result_store_mutex.lock();

  charge_results[thread_id] = annealer->db_charges;
  energy_results[thread_id] = annealer->config_energies;
  //object->numElecStore[threadId] = object->n_elec;

  result_store_mutex.unlock();
}

float SimAnneal::distance(const int &i, const int &j)
{
  int x1 = sim_params.db_locs[i].first;
  int y1 = sim_params.db_locs[i].second;
  int x2 = sim_params.db_locs[j].first;
  int y2 = sim_params.db_locs[j].second;
  return sqrt(pow(x1-x2, 2.0) + pow(y1-y2, 2.0));
}

float SimAnneal::interElecPotential(const float &r)
{
  //return exp(-r/debye_length) / r;
  //return constants::Q0 * Kc * erf(r/constants::ERFDB) * exp(-r/debye_length) / r;
  return constants::Q0 * sim_params.Kc * exp(-r/sim_params.debye_length) / r;
}



// SimAnnealThread Implementation

SimAnnealThread::SimAnnealThread(const int t_thread_id)
  : thread_id(t_thread_id)
{
  rng.seed(std::time(NULL)*thread_id+4065);
  dis01 = boost::random::uniform_real_distribution<float>(0,1);
}

void SimAnnealThread::run()
{
  // initialize variables & perform pre-calculation
  kT = sparams->T_init*constants::Kb;
  v_freeze = 0;

  // resize vectors
  v_local.resize(sparams->n_dbs);

  db_charges.resize(sparams->result_queue_size);
  n.resize(sparams->n_dbs);
  occ.resize(sparams->n_dbs);

  config_energies.resize(sparams->result_queue_size);

  // SIM ANNEAL
  anneal();
}

void SimAnnealThread::anneal()
{
  // Vars
  boost::numeric::ublas::vector<int> dn(sparams->n_dbs); // change of occupation for population update
  int occ_ind_min, occ_ind_max, unocc_ind_min, unocc_ind_max;
  int from_occ_ind, to_occ_ind; // hopping from n[occ[from_ind]]
  int from_ind, to_ind;         // hopping from n[from_ind] to n[to_ind]
  int hop_attempts;
  bool pop_changed;

  E_sys = systemEnergy();
  v_local = sparams->v_ext - ublas::prod(sparams->v_ij, n);

  // Run simulated annealing for predetermined time steps
  while(t < sparams->anneal_cycles) {
    //std::cout << "Cycle " << t << ", kT=" << kT << ", v_freeze=" << v_freeze << std::endl;

    // Random population change, pop_changed is set to true if anything changes
    dn = genPopDelta(pop_changed);
    if (pop_changed) {
      n += dn;
      E_sys += -1 * ublas::inner_prod(v_local, dn) + totalCoulombPotential(dn);
      v_local -= ublas::prod(sparams->v_ij, dn);

      // Occupation list update (used for selecting sites to hop from and to)
      // First n_dbs entries are occupied, the rest are unoccupied
      int occ_ind=0, unocc_ind=sparams->n_dbs-1;
      for (int db_ind=0; db_ind<sparams->n_dbs; db_ind++) {
        if (n[db_ind])
          occ[occ_ind++] = db_ind;
        else
          occ[unocc_ind--] = db_ind;
      }
      n_elec = occ_ind;
      occ_ind_min = 0;
      occ_ind_max = n_elec-1;
      unocc_ind_min = n_elec;
      unocc_ind_max = sparams->n_dbs-1;
    }

    // Hopping - randomly hop electrons from occupied sites to unoccupied sites
    hop_attempts = 0;
    if (n_elec != 0) {
      while (hop_attempts < (sparams->n_dbs-n_elec)*5) {
        from_occ_ind = randInt(occ_ind_min, occ_ind_max);
        to_occ_ind = randInt(unocc_ind_min, unocc_ind_max);
        from_ind = occ[from_occ_ind];
        to_ind = occ[to_occ_ind];

        float E_del = hopEnergyDelta(from_ind, to_ind);
        if (acceptHop(E_del)) {
          performHop(from_ind, to_ind);
          occ[from_occ_ind] = to_ind;
          occ[to_occ_ind] = from_ind;
          // calculate energy difference
          E_sys += E_del;
          ublas::matrix_column<ublas::matrix<float>> v_i (sparams->v_ij, from_ind);
          ublas::matrix_column<ublas::matrix<float>> v_j (sparams->v_ij, to_ind);
          v_local += v_i - v_j;
        }
        hop_attempts++;
      }
    }

    // push back the new arrangement
    db_charges.push_back(n);
    config_energies.push_back(E_sys);

    /*
    std::cout << "db_charges=";
    for (int charge : n)
      std::cout << charge;
    std::cout << ", system energy=" << systemEnergy();
    std::cout << std::endl;
    */

    // perform time-step if not pre-annealing
    timeStep();
  }

  SimAnneal::storeResults(this, thread_id);
}

ublas::vector<int> SimAnnealThread::genPopDelta(bool &changed)
{
  changed = false;
  ublas::vector<int> dn(sparams->n_dbs);
  for (unsigned i=0; i<n.size(); i++) {
    float prob = 1. / ( 1 + exp( ((2*n[i]-1)*(v_local[i] + sparams->mu) + v_freeze) / kT ) );

    if (evalProb(prob)) {
      dn[i] = 1 - 2*n[i];
      changed = true;
    } else {
      dn[i] = 0;
    }
  }
  return dn;
}

void SimAnnealThread::performHop(const int &from_ind, const int &to_ind)
{
  n[from_ind] = 0;
  n[to_ind] = 1;
}

void SimAnnealThread::timeStep()
{
  t++;
  if (t > sparams->preanneal_cycles) {
    kT = sparams->kT_min + (kT - sparams->kT_min) * sparams->alpha;
    v_freeze = (t - sparams->preanneal_cycles) * sparams->v_freeze_step;
  }
}

bool SimAnnealThread::acceptHop(const float &v_diff)
{
  if (v_diff < 0)
    return true;

  // some acceptance function, acceptance probability falls off exponentially
  float prob = exp(-v_diff/kT);

  return evalProb(prob);
}

bool SimAnnealThread::evalProb(const float &prob)
{
  boost::variate_generator<boost::random::mt19937&, boost::random::uniform_real_distribution<float>> rnd_gen(rng, dis01);
  return prob >= rnd_gen();
}

int SimAnnealThread::randInt(const int &min, const int &max)
{
  boost::random::uniform_int_distribution<int> dis(min,max);
  return dis(rng);
}

float SimAnnealThread::systemEnergy() const
{
  assert(sparams->n_dbs > 0);
  float v = 0;
  for(int i=0; i<sparams->n_dbs; i++) {
    v -= sparams->v_ext[i] * n[i];
    for(int j=i+1; j<sparams->n_dbs; j++)
      v += sparams->v_ij(i,j) * n[i] * n[j];
  }
  return v;
}

float SimAnnealThread::totalCoulombPotential(ublas::vector<int> &config) const
{
  return 0.5 * ublas::inner_prod(config, ublas::prod(sparams->v_ij, config));
}

float SimAnnealThread::hopEnergyDelta(const int &i, const int &j)
{
  return v_local[i] - v_local[j] - sparams->v_ij(i,j);
}
