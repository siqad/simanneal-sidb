// @file:     sim_anneal.cc
// @author:   Samuel
// @created:  2017.08.23
// @editted:  2019.02.01
// @license:  Apache License 2.0
//
// @desc:     Simulated annealing physics engine

#include "charge_domains.h"
#include "affinity_workers.h"
#include "population_blas.h"
#include "simanneal.h"
#include <new>
#ifdef SIMANNEAL_HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif
#include <ctime>
#include <algorithm>
#include <unordered_set>
#include <atomic>
#include <exception>
#include <future>
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
std::recursive_mutex SimAnneal::active_model_mutex;
bool SimAnneal::model_active_ = false;
std::unique_ptr<const simanneal_pair_bound::Geometry>
    SimAnneal::repair_geometry_;
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
  if (db_locs.size() >
      static_cast<std::size_t>((std::numeric_limits<int>::max() - 8) / 4))
    throw std::invalid_argument("Too many DB sites for bounded work counters");
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

void SimParams::setFixedCharges(const std::vector<EuclCoord3d> &locations,
                                const std::vector<FPType> &charges,
                                const std::vector<FPType> &permittivities,
                                const std::vector<FPType> &screening_lengths) {
  const auto count = locations.size();
  if (charges.size() != count || permittivities.size() != count ||
      screening_lengths.size() != count)
    throw std::invalid_argument(
        "Fixed-charge parameter vectors must have equal lengths");
  if (db_locs.empty() || v_fc.size() != db_locs.size())
    throw std::invalid_argument("Set DB locations before fixed charges");
  for (std::size_t j = 0; j < count; ++j) {
    const auto &location = locations[j];
    if (!std::isfinite(location.x) || !std::isfinite(location.y) ||
        !std::isfinite(location.z) || !std::isfinite(charges[j]) ||
        !std::isfinite(permittivities[j]) || permittivities[j] <= 0 ||
        !std::isfinite(screening_lengths[j]) || screening_lengths[j] <= 0)
      throw std::invalid_argument(
          "Fixed charges require finite coordinates/charge and positive finite "
          "physical parameters");
  }
  ublas::vector<FPType> potential(db_locs.size());
  potential.clear();
  for (std::size_t i = 0; i < db_locs.size(); ++i)
    for (std::size_t j = 0; j < count; ++j) {
      const double radius =
          SimAnneal::distance(db_locs[i].first, db_locs[i].second, 0,
                              locations[j].x, locations[j].y, locations[j].z) *
          SimAnneal::db_distance_scale;
      if (!(radius > 0) || !std::isfinite(radius))
        throw std::invalid_argument(
            "Fixed charge distance must be finite and positive");
      potential[i] += SimAnneal::coulombicPotential(
          charges[j], 1, permittivities[j], screening_lengths[j], radius);
      if (!std::isfinite(potential[i]))
        throw std::invalid_argument(
            "Fixed-charge potential must remain finite");
    }
  v_fc.swap(potential); // Failed validation never replaces the prior field.
}

// SimAnneal (master) Implementation

SimAnneal::SimAnneal(SimParams &parameters) : SimAnneal(parameters, false) {}
SimAnneal::SimAnneal(SimParams &&parameters) : SimAnneal(parameters, true) {}

SimAnneal::SimAnneal(SimParams &parameters, bool consume) {
  {
    std::lock_guard<std::recursive_mutex> lock(active_model_mutex);
    if (model_active_)
      throw std::logic_error("Only one SimAnneal model may be alive at a time");
    model_active_ = true;
  }
  try {
    if (consume) {
      // uBLAS matrix move assignment can copy; swap guarantees storage transfer.
      ublas::matrix<FPType> distances, coupling;
      distances.swap(parameters.db_r);
      coupling.swap(parameters.v_ij);
      sim_params = std::move(parameters);
      sim_params.db_r.swap(distances);
      sim_params.v_ij.swap(coupling);
    } else {
      sim_params = parameters;
    }
    initialize();
  } catch (...) {
    repair_geometry_.reset();
    sim_params = SimParams();
    std::lock_guard<std::recursive_mutex> lock(active_model_mutex);
    model_active_ = false;
    throw;
  }
}

SimAnneal::~SimAnneal() {
  for (auto &worker : anneal_threads)
    if (worker.joinable())
      worker.join();
  refinement_geometry_.reset();
  repair_geometry_.reset();
  AllChargeResults().swap(charge_results);
  AllEnergyResults().swap(energy_results);
  SuggestedResults().swap(suggested_gs_results);
  sim_params = SimParams();
  std::lock_guard<std::recursive_mutex> lock(active_model_mutex);
  model_active_ = false;
}

RepairResult
SimAnneal::repairConfiguration(const ublas::vector<int> &configuration,
                               bool known_initial_invalid) {
  return boundedPhysicalRepair(sim_params, configuration,
                               repair_geometry_.get(), known_initial_invalid);
}

void SimAnneal::invokeSimAnneal()
{
  std::unique_lock<std::recursive_mutex> invocation(invocation_mutex_,
                                                    std::try_to_lock);
  if (!invocation.owns_lock() || invocation_active_)
    throw std::logic_error("SimAnneal invocation is already active");
  invocation_active_ = true;
  struct InvocationReset {
    bool &active;
    ~InvocationReset() { active = false; }
  } reset{invocation_active_};
  // Proof belongs solely to this invocation. Workers join before destruction.
  struct SearchProofScope {
    bool finite;
    ~SearchProofScope() { finite = false; }
  } search_scope{currentSearchModelProof()};
  const simanneal_blas::ScopedThreadCount blas_threads(
      1, sim_params.population_backend == PopulationBackend::OpenBLAS ||
             sim_params.population_backend == PopulationBackend::OpenBLASSymmetric);
  stats_ = SearchStats();
  stats_.population_blas_threads =
      (sim_params.population_backend == PopulationBackend::OpenBLAS ||
       sim_params.population_backend == PopulationBackend::OpenBLASSymmetric)
          ? simanneal_blas::threadCount()
          : 0;
  if (refinement_geometry_) {
    stats_.refinement_center_offset = refinement_geometry_->centerOffset();
    stats_.refinement.geometry_count = refinement_geometry_->geometryCount();
    stats_.refinement.shared_single_cache_fallback =
        refinement_geometry_->sharedSingleCacheFallback();
  }
  if (sim_params.singleton_enabled) {
    bool all = true;
    ublas::vector<int> configuration(sim_params.n_dbs);
    for (int i = 0; i < sim_params.n_dbs; ++i) {
      all &= simanneal_domains::singleton(sim_params.final_domains[i]);
      configuration[i] =
          simanneal_domains::singletonCharge(sim_params.final_domains[i]);
    }
    FPType energy;
    if (all && validatedSearchEnergy(configuration, energy, search_scope.finite)) {
      charge_results.clear();
      energy_results.clear();
      suggested_gs_results.assign(
          1, ChargeConfigResult(configuration, true, energy));
      stats_.singleton_used = true;
      return;
    }
  }
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
        SimAnnealThread annealer(id, seeds[id], search_scope.finite);
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
  stats_.executed_restarts = sim_params.num_instances;
  for (const auto &candidate : suggested_gs_results) {
    stats_.repair_attempts += candidate.repair_attempted;
    stats_.repair_budget_exhaustions += candidate.repair_budget_exhausted;
  }
  if (refinement_geometry_) {
    const refinement::ModelView model(sim_params.v_ij, sim_params.v_ext,
                                      sim_params.v_fc, sim_params.final_domains,
                                      sim_params.mu, constants::eta,
                                      constants::RECALC_STABILITY_ERR);
    std::vector<refinement::Candidate> candidates;
    for (const auto &candidate : suggested_gs_results)
      if (candidate.initialized)
        candidates.emplace_back(candidate.config, candidate.system_energy,
                                true);
    // Refinement owns one normalization pass. Public endpoint flags/energies
    // are never used as a physical proof or as the normalized ranking.
    refinement::Callbacks callbacks;
    const bool finite = search_scope.finite;
    callbacks.validate = [finite](const ublas::vector<int> &q, FPType &e) {
      return SimAnneal::validatedSearchEnergy(q, e, finite);
    };
    callbacks.repair = [](const ublas::vector<int> &q) {
      const auto r = SimAnneal::repairConfiguration(q);
      refinement::Candidate candidate(r.config, r.energy, r.valid);
      candidate.budget_exhausted = r.budget_exhausted;
      return candidate;
    };
    const auto result = refinement::run(
        model, *refinement_geometry_, candidates, sim_params.refinement_options,
        std::min(sim_params.num_workers,
                 sim_params.refinement_options.candidates),
        callbacks);
    stats_.refinement = result.stats;
    if (result.improved && result.candidate.valid) {
      FPType energy;
      if (validatedSearchEnergy(result.candidate.config, energy, search_scope.finite)) {
        suggested_gs_results.emplace_back(result.candidate.config, true,
                                          energy);
        suggested_gs_results.back().refinement_result = true;
      }
    }
  }

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

bool SimAnneal::evaluateConfiguration(const ublas::vector<int> &charge,
                                      FPType *energy) {
  const auto &sp = sim_params;
  const std::size_t size = charge.size();
  if (size == 0 || size != static_cast<std::size_t>(sp.n_dbs) ||
      sp.v_ext.size() != size || sp.v_fc.size() != size ||
      sp.v_ij.size1() != size || sp.v_ij.size2() != size)
    return false;
  const FPType eps = constants::RECALC_STABILITY_ERR;
  thread_local ublas::vector<FPType> potential;
  if (potential.size() != size) potential.resize(size, false);
  for (std::size_t i = 0; i < size; ++i) {
    if (charge[i] < -1 || charge[i] > 1 || sp.v_ij(i, i) != 0)
      return false;
    potential[i] = -(sp.v_ext[i] + sp.v_fc[i]);
    const FPType *row = &sp.v_ij.data()[i * size];
    for (std::size_t j = 0; j < i; ++j)
      potential[i] -= row[j] * charge[j];
    for (std::size_t j = i + 1; j < size; ++j)
      potential[i] -= row[j] * charge[j];
    if (!std::isfinite(potential[i]))
      return false;
    const FPType value = potential[i] + sp.mu;
    const FPType upper_value = potential[i] + (sp.mu - constants::eta);
    if (!((charge[i] == -1 && value < eps) ||
          (charge[i] == 1 && upper_value > -eps) ||
          (charge[i] == 0 && value > -eps && upper_value < eps)))
      return false;
  }
  // Final acceptance never trusts cached proposal geometry. Rebuild every
  // allowed ordered hop against the current full physical model.
  std::vector<std::size_t> above_negative, above_neutral;
  above_negative.reserve(size); above_neutral.reserve(size);
  for (std::size_t j = 0; j < size; ++j) {
    if (charge[j] > -1) above_negative.push_back(j);
    if (charge[j] > 0) above_neutral.push_back(j);
  }
  for (std::size_t i = 0; i < size; ++i) {
    if (charge[i] == 1) continue;
    const auto &targets = charge[i] == -1 ? above_negative : above_neutral;
    for (std::size_t j : targets)
      if (-potential[i] + potential[j] - sp.v_ij(i,j) < -eps) return false;
  }
  if (energy) {
    FPType sum = 0;
    for (std::size_t i = 0; i < size; ++i)
      sum += charge[i] * (sp.v_ext[i] + sp.v_fc[i] - potential[i]);
    sum *= 0.5;
    // The fused expression can overflow an intermediate for extreme finite
    // fields even when the original Hamiltonian remains representable.
    if (!std::isfinite(sum))
      sum = systemEnergy(charge);
    if (!std::isfinite(sum))
      return false;
    *energy = sum;
  }
  return true;
}

bool SimAnneal::currentSearchModelProof() {
  const auto &sp = sim_params;
  if (sp.n_dbs <= 0) return false;
  const std::size_t size = static_cast<std::size_t>(sp.n_dbs);
  if (sp.v_ext.size() != size || sp.v_fc.size() != size ||
      sp.v_ij.size1() != size || sp.v_ij.size2() != size ||
      !std::isfinite(sp.mu)) return false;
  for (std::size_t i = 0; i < size; ++i) {
    if (!std::isfinite(sp.v_ext[i]) || !std::isfinite(sp.v_fc[i]) ||
        sp.v_ij(i,i) != 0) return false;
    const FPType *row = &sp.v_ij.data()[i * size];
    for (std::size_t j = 0; j < size; ++j)
      if (!std::isfinite(row[j])) return false;
  }
  return true;
}
bool SimAnneal::validatedSearchEnergy(const ublas::vector<int> &q,
                                      FPType &energy, bool finite_model) {
  return evaluateSearchConfiguration(q, &energy, finite_model);
}
bool SimAnneal::evaluateSearchConfiguration(const ublas::vector<int> &charge,
                                      FPType *energy, bool finite_model) {
  if (!finite_model) return evaluateConfiguration(charge, energy);
  const auto &sp = sim_params;
  const std::size_t size = charge.size();
  if (size == 0 || size != static_cast<std::size_t>(sp.n_dbs) ||
      sp.v_ext.size() != size || sp.v_fc.size() != size ||
      sp.v_ij.size1() != size || sp.v_ij.size2() != size)
    return false;
  const FPType eps = constants::RECALC_STABILITY_ERR;
  thread_local ublas::vector<FPType> potential;
  if (potential.size() != size) potential.resize(size, false);
  thread_local std::vector<std::size_t> charged;
  charged.clear(); charged.reserve(size);
  for (std::size_t i = 0; i < size; ++i) {
    if (charge[i] < -1 || charge[i] > 1 || sp.v_ij(i, i) != 0)
      return false;
    potential[i] = -(sp.v_ext[i] + sp.v_fc[i]);
    const FPType *row = &sp.v_ij.data()[i * size];
    if (i == 0) {
      // Build the ascending charged list while computing the first row.
      if (charge[0] != 0) charged.push_back(0);
      for (std::size_t j = 1; j < size; ++j)
        if (charge[j] != 0) {
          charged.push_back(j);
          potential[i] -= row[j] * charge[j];
        }
    } else {
      for (std::size_t j : charged)
        if (j != i) potential[i] -= row[j] * charge[j];
    }
    if (!std::isfinite(potential[i]))
      return false;
    const FPType value = potential[i] + sp.mu;
    const FPType upper_value = potential[i] + (sp.mu - constants::eta);
    if (!((charge[i] == -1 && value < eps) ||
          (charge[i] == 1 && upper_value > -eps) ||
          (charge[i] == 0 && value > -eps && upper_value < eps)))
      return false;
  }
  // Final acceptance never trusts cached proposal geometry. Rebuild every
  // allowed ordered hop against the current full physical model.
  std::vector<std::size_t> above_negative, above_neutral;
  above_negative.reserve(size); above_neutral.reserve(size);
  for (std::size_t j = 0; j < size; ++j) {
    if (charge[j] > -1) above_negative.push_back(j);
    if (charge[j] > 0) above_neutral.push_back(j);
  }
  for (std::size_t i = 0; i < size; ++i) {
    if (charge[i] == 1) continue;
    const auto &targets = charge[i] == -1 ? above_negative : above_neutral;
    for (std::size_t j : targets)
      if (-potential[i] + potential[j] - sp.v_ij(i,j) < -eps) return false;
  }
  if (energy) {
    FPType sum = 0;
    for (std::size_t i = 0; i < size; ++i)
      sum += charge[i] * (sp.v_ext[i] + sp.v_fc[i] - potential[i]);
    sum *= 0.5;
    // The fused expression can overflow an intermediate for extreme finite
    // fields even when the original Hamiltonian remains representable.
    if (!std::isfinite(sum))
      sum = systemEnergy(charge);
    if (!std::isfinite(sum))
      return false;
    *energy = sum;
  }
  return true;
}

bool SimAnneal::isMetastable(const ublas::vector<int> &charge) {
  return evaluateConfiguration(charge, nullptr);
}
bool SimAnneal::validatedEnergy(const ublas::vector<int> &charge,
                                FPType &energy) {
  return evaluateConfiguration(charge, &energy);
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
  export_workers_ = 1;
  export_async_launches_ = 0;
  if (tidy) {
    std::unordered_set<std::string> config_set;
    std::vector<const ChargeConfigResult *> unique;
    unique.reserve(suggested_gs_results.size());
    for (const auto &result : suggested_gs_results)
      if (result.initialized &&
          config_set.insert(configToStr(result.config)).second)
        unique.push_back(&result);

    std::vector<FPType> energies(unique.size());
    // Distinct bytes avoid the shared-bit writes of vector<bool>.
    std::vector<unsigned char> valid(unique.size(), 0);
    const auto validate_range = [&](std::size_t begin, std::size_t end) {
      for (std::size_t i = begin; i < end; ++i)
        valid[i] = validatedEnergy(unique[i]->config, energies[i]);
    };
    const std::size_t worker_limit = std::max(1, sim_params.num_workers);
    const std::size_t workers = sim_params.n_dbs >= 128 &&
        double(unique.size()) * sim_params.n_dbs * sim_params.n_dbs >= 4e6
        ? std::min(worker_limit, (unique.size() + 7) / 8) : 1;
    export_workers_ = workers;
    if (workers == 1) {
      validate_range(0, unique.size());
    } else {
      // Futures die before the buffers and lambda, also during unwinding.
      std::vector<std::future<void>> tasks;
      tasks.reserve(workers);
      std::size_t assigned = 0;
      try {
        for (std::size_t worker = 0; worker < workers; ++worker) {
          const std::size_t begin = unique.size() * worker / workers;
          const std::size_t end = unique.size() * (worker + 1) / workers;
          tasks.emplace_back(std::async(std::launch::async, validate_range,
                                       begin, end));
          assigned = end;
          ++export_async_launches_;
        }
      } catch (const std::system_error &) {
        // Already launched tasks own the prefix. Complete only the tail.
        validate_range(assigned, unique.size());
      }
      for (auto &task : tasks)
        task.get();
    }
    filtered_results.reserve(unique.size());
    for (std::size_t i = 0; i < unique.size(); ++i)
      if (valid[i]) {
        filtered_results.push_back(*unique[i]);
        filtered_results.back().system_energy = energies[i];
      }
  } else {
    for (const auto &result : suggested_gs_results)
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
      sp.population_backend != PopulationBackend::Accelerate &&
      sp.population_backend != PopulationBackend::OpenBLAS &&
      sp.population_backend != PopulationBackend::OpenBLASSymmetric)
    throw std::invalid_argument("Unknown population_backend");

  if (sp.search_profile != SearchProfile::Legacy &&
      sp.search_profile != SearchProfile::Optimized)
    throw std::invalid_argument("Unknown search_profile");
  if (sp.random_backend == RandomBackend::Auto)
    sp.random_backend = sp.search_profile == SearchProfile::Optimized
                            ? RandomBackend::PCG32
                            : RandomBackend::MT;
  if (sp.random_backend != RandomBackend::MT &&
      sp.random_backend != RandomBackend::PCG32)
    throw std::invalid_argument("Unknown random_backend");
  const auto enabled = [&](FeatureSetting setting) {
    if (setting != FeatureSetting::ProfileDefault &&
        setting != FeatureSetting::Disabled &&
        setting != FeatureSetting::Enabled)
      throw std::invalid_argument("Unknown feature setting");
    return setting == FeatureSetting::Enabled ||
           (setting == FeatureSetting::ProfileDefault &&
            sp.search_profile == SearchProfile::Optimized);
  };
  sp.repair_enabled = enabled(sp.repair);
  sp.singleton_enabled = enabled(sp.singleton_shortcut);
  if (sp.n_dbs <= 0 ||
      sp.db_locs.size() != static_cast<std::size_t>(sp.n_dbs) ||
      sp.v_ext.size() != static_cast<std::size_t>(sp.n_dbs) ||
      sp.v_fc.size() != static_cast<std::size_t>(sp.n_dbs))
    throw std::invalid_argument("Invalid model dimensions");
  if (!std::isfinite(sp.mu) || !std::isfinite(sp.eps_r) || sp.eps_r <= 0 ||
      !std::isfinite(sp.debye_length) || sp.debye_length <= 0)
    throw std::invalid_argument("Invalid physical parameters");
  for (int i = 0; i < sp.n_dbs; ++i)
    if (!std::isfinite(sp.db_locs[i].first) ||
        !std::isfinite(sp.db_locs[i].second) || !std::isfinite(sp.v_ext[i]) ||
        !std::isfinite(sp.v_fc[i]))
      throw std::invalid_argument("Coordinates and fields must be finite");
  if (!std::isfinite(sp.result_queue_factor) || sp.result_queue_factor < 0 ||
      sp.result_queue_factor > 1)
    throw std::invalid_argument("result_queue_factor must be in [0,1]");
  if (sp.anneal_cycles <= 0 || sp.preanneal_cycles < 0 ||
      sp.preanneal_cycles > sp.anneal_cycles || sp.hop_attempt_factor < 0 ||
      !std::isfinite(sp.T_init) || sp.T_init <= 0 || !std::isfinite(sp.T_min) ||
      sp.T_min <= 0 || !std::isfinite(sp.T_e_inv_point) ||
      sp.T_e_inv_point <= 0 || !std::isfinite(sp.v_freeze_end_point) ||
      sp.v_freeze_end_point <= 0 || !std::isfinite(sp.v_freeze_threshold) ||
      sp.v_freeze_threshold <= 0 || sp.phys_validity_check_cycles <= 0)
    throw std::invalid_argument("Invalid annealing schedule");
  if (sp.population_backend == PopulationBackend::OpenBLAS ||
      sp.population_backend == PopulationBackend::OpenBLASSymmetric) {
    if (!simanneal_blas::openblasAvailable())
      throw std::invalid_argument("OpenBLAS is unavailable in this build");
  }
  if (sp.deterministic_seed && sp.random_seed > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("random_seed must be in [0,4294967295]");

  if (saglobal::log_level >= Logger::DBG) log.debug() << "Performing pre-calculations..." << std::endl;

  if (!std::isfinite(sp.v_freeze_init) || !std::isfinite(sp.v_freeze_reset))
    throw std::invalid_argument(
        "Freeze initial/reset potentials must be finite");
  if (sp.hop_attempt_factor > std::numeric_limits<int>::max() / sp.n_dbs)
    throw std::invalid_argument("Hop attempt budget exceeds supported range");

  // set default values
  if (sp.v_freeze_init < 0)
    sp.v_freeze_init = fabs(sp.mu) / 2;
  if (sp.v_freeze_reset < 0)
    sp.v_freeze_reset = fabs(sp.mu);

  // apply schedule scaling
  sp.alpha = std::pow(std::exp(-1.), 1./(sp.T_e_inv_point * sp.anneal_cycles));
  const double freeze_cycles = sp.v_freeze_end_point * sp.anneal_cycles;
  if (!std::isfinite(freeze_cycles) ||
      freeze_cycles >
          std::numeric_limits<int>::max() - sp.phys_validity_check_cycles)
    throw std::invalid_argument(
        "Freeze schedule exceeds supported cycle range");
  sp.v_freeze_cycles = freeze_cycles < 1 ? 1 : static_cast<int>(freeze_cycles);
  sp.v_freeze_step = sp.v_freeze_threshold / sp.v_freeze_cycles;

  if (saglobal::log_level >= Logger::DBG) log.debug() << "Anneal cycles: " << sp.anneal_cycles << ", alpha: "
    << sp.alpha << ", v_freeze_cycles: " << sp.v_freeze_cycles << std::endl;

  sp.result_queue_size = static_cast<int>(
      static_cast<double>(sp.anneal_cycles) * sp.result_queue_factor);
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
  sp.num_workers = simanneal_affinity::workerCount(sp.num_workers, sp.num_instances);
  // inter-db distances and voltages
  sp.population_finite_matrix = true;
  const int geometry_workers = simanneal_affinity::geometryWorkerCount(
      sp.num_workers, sp.n_dbs, saglobal::log_level >= Logger::DBG);
  if (geometry_workers > 1) {
    // Each unordered pair has one writer and retains the serial arithmetic.
    std::vector<unsigned char> finite(sp.n_dbs, 1);
    std::atomic<int> next_row(0);
    std::exception_ptr failure;
    std::mutex failure_mutex;
    const auto work = [&] {
      try {
        for (;;) {
          const int first = next_row.fetch_add(8);
          if (first >= sp.n_dbs) break;
          for (int i = first; i < std::min(first + 8, sp.n_dbs); ++i) {
            sp.db_r(i,i) = 0.;
            sp.v_ij(i,i) = 0.;
            for (int j = i + 1; j < sp.n_dbs; ++j) {
              sp.db_r(i,j) = db_distance_scale * distance(i,j);
              sp.v_ij(i,j) = interElecPotential(sp.db_r(i,j));
              if (!std::isfinite(sp.v_ij(i,j))) finite[i] = 0;
              sp.db_r(j,i) = sp.db_r(i,j);
              sp.v_ij(j,i) = sp.v_ij(i,j);
            }
          }
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(failure_mutex);
        if (!failure) failure = std::current_exception();
      }
    };
    std::vector<std::thread> threads;
    try {
      for (int i = 1; i < geometry_workers; ++i) threads.emplace_back(work);
    } catch (...) {
      for (auto &thread : threads) thread.join();
      throw;
    }
    work();
    for (auto &thread : threads) thread.join();
    if (failure) std::rethrow_exception(failure);
    for (const auto value : finite) sp.population_finite_matrix &= value != 0;
  } else {
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

  const auto &options = sp.refinement_options;
  if (options.mode != refinement::Mode::Disabled &&
      options.mode != refinement::Mode::K6 &&
      options.mode != refinement::Mode::K10 &&
      options.mode != refinement::Mode::SharedK10)
    throw std::invalid_argument("Unknown refinement mode");
  if (options.candidates < 1 || options.candidates > 32 || options.rounds < 1 ||
      options.rounds > 8 || options.trials < 1 || options.trials > 2 ||
      options.geometry_byte_cap == 0 ||
      options.geometry_byte_cap > 8 * 1024 * 1024 ||
      options.dedup_byte_cap == 0 || options.dedup_byte_cap > 8 * 1024 * 1024)
    throw std::invalid_argument("Invalid bounded refinement settings");
  sp.final_domains = simanneal_domains::finalCharges(
      sp.v_ij, sp.v_ext, sp.v_fc, sp.n_dbs, sp.mu, constants::eta,
      std::max(constants::POP_STABILITY_ERR, constants::RECALC_STABILITY_ERR));
  if (sp.population_finite_matrix && sp.n_dbs >= 256 &&
      (sp.repair_enabled ||
       sp.refinement_options.mode != refinement::Mode::Disabled))
    repair_geometry_.reset(
        new simanneal_pair_bound::Geometry(sp.n_dbs, sp.v_ij));
  if (sp.refinement_options.mode != refinement::Mode::Disabled) {
    const refinement::ModelView model(sp.v_ij, sp.v_ext, sp.v_fc,
                                      sp.final_domains, sp.mu, constants::eta,
                                      constants::RECALC_STABILITY_ERR);
    std::uint64_t geometry_seed = sp.random_seed;
    if (!sp.deterministic_seed) {
      boost::random_device entropy;
      geometry_seed = (static_cast<std::uint64_t>(entropy()) << 32) | entropy();
    }
    refinement_geometry_.reset(
        new refinement::Geometry(model, sp.refinement_options, geometry_seed));
  }
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

SimAnnealThread::SimAnnealThread(const int id, const std::uint64_t seed)
    : SimAnnealThread(id, seed, false) {}
SimAnnealThread::SimAnnealThread(const int id, const std::uint64_t seed,
                               bool finite_model)
    : thread_id(id), search_finite_model_(finite_model), dis01(0, 1) {
  if (sparams->n_dbs <= 0 || id < 0 || id >= sparams->num_instances)
    throw std::invalid_argument(
        "Worker requires a live model and valid restart ID");
  use_pcg = sparams->random_backend == RandomBackend::PCG32;
  if (use_pcg)
    new (&engines.pcg) PcgEngine(seed);
  else
    new (&engines.mt) RandEng(seed);
  muzm = sparams->mu;
  mupz = sparams->mu - constants::eta;
}

SimAnnealThread::~SimAnnealThread() {
  if (use_pcg)
    engines.pcg.~PcgEngine();
  else
    engines.mt.~RandEng();
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
  const bool population_openblas =
      sparams->population_backend == PopulationBackend::OpenBLAS ||
      sparams->population_backend == PopulationBackend::OpenBLASSymmetric;
#ifdef SIMANNEAL_HAVE_ACCELERATE
  const bool population_accelerate =
      sparams->population_backend == PopulationBackend::Auto ||
      sparams->population_backend == PopulationBackend::Accelerate;
#else
  const bool population_accelerate = false;
#endif
  std::vector<FPType> population_backend_input(
      population_accelerate || population_openblas ? sparams->n_dbs : 0);
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

  if (sparams->transient_domain_mask) {
    for (int i = 0; i < sparams->n_dbs; ++i) {
      const auto mask = sparams->final_domains[i];
      if (simanneal_domains::singleton(mask))
        n[i] = simanneal_domains::singletonCharge(mask);
    }
    for (int i = 0; i < sparams->n_dbs; ++i) {
      if (n[i] == -1)
        dbm_occ[dbm_occ_count++] = i;
      else if (n[i] == 1)
        dbp_occ[dbp_occ_count++] = i;
      else {
        if (local_hops)
          neutral_slot[i] = db0_occ_count;
        db0_occ[db0_occ_count++] = i;
      }
    }
  }
  E_sys = systemEnergy();
  v_local = - (sparams->v_ext + sparams->v_fc) - ublas::prod(sparams->v_ij, n);

  Logger log(saglobal::log_level);

  bool validity_cached = false, cached_validity = false;
  // Run simulated annealing for predetermined time steps
  while(t < sparams->anneal_cycles) {
    //log.debug() << "Cycle " << t << ", kT=" << kT << ", v_freeze=" << v_freeze << std::endl;

    // Random population change, pop_changed is set to true if anything changes
    //log.debug() << "Before popgen: n=" << n << std::endl;
    genPopDelta(dn, pop_changed);
    if (pop_changed) {
      validity_cached = false;
      n += dn;
      if (radius_hops) population_changed.clear();
      bool dense_updated = false;
      int population_nonzero = 0;
      if (population_openblas)
        for (unsigned j = 0; j < dn.size(); ++j)
          population_nonzero += dn[j] != 0;
      const bool openblas_dense =
          population_openblas && population_nonzero * 4 >= sparams->n_dbs;
      if ((population_accelerate || openblas_dense) &&
          population_finite_matrix && !dn.empty()) {
        const int count = static_cast<int>(dn.size());
        for (int j = 0; j < count; ++j) {
          population_backend_input[j] = dn[j];
          if (radius_hops && dn[j] != 0)
            population_changed.push_back(static_cast<unsigned>(j));
        }
        FPType *output = &population_v_delta.data()[0];
        if (population_openblas) {
          dense_updated =
              sparams->population_backend ==
                      PopulationBackend::OpenBLASSymmetric
                  ? simanneal_blas::symv(&sparams->v_ij.data()[0],
                                         population_backend_input.data(),
                                         output, count)
                  : simanneal_blas::gemv(&sparams->v_ij.data()[0],
                                         population_backend_input.data(),
                                         output, count);
        }
#ifdef SIMANNEAL_HAVE_ACCELERATE
        else {
          cblas_dgemv(CblasRowMajor, CblasNoTrans, count, count, 1.0,
                      &sparams->v_ij.data()[0], count,
                      population_backend_input.data(), 1, 0.0, output, 1);
          dense_updated = true;
        }
#endif
        if (dense_updated) {
          FPType linear = 0, quadratic = 0;
          for (int i = 0; i < count; ++i) {
            linear += v_local[i] * dn[i];
            quadratic += dn[i] * output[i];
          }
          E_sys += -linear + 0.5 * quadratic;
          for (int i = 0; i < count; ++i)
            v_local[i] -= output[i];
        }
      }
      if (!dense_updated) {
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
                          uniformDraw() >= sparams->hop_global_probability))) {
        to_ind = radius_hops ? sparams->hop_neighborhood.selectRadius(
                                   from_ind, n, uniformDraw(), radius_cache)
                             : sparams->hop_neighborhood.select(from_ind, n,
                                                                uniformDraw());
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
      const int hop_direction = n[from_ind] == -1 ? 1 : -1;
      if (sparams->transient_domain_mask &&
          (!(sparams->final_domains[from_ind] &
             (1 << (n[from_ind] + hop_direction + 1))) ||
           !(sparams->final_domains[to_ind] &
             (1 << (n[to_ind] - hop_direction + 1))))) {
        ++hop_attempts;
        continue;
      }
      if (acceptHop(hop_E_del)) {
        validity_cached = false;
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
    if (!validity_cached) { cached_validity = populationValid(); validity_cached = true; }
    const bool pop_valid = cached_validity;
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
  if (sparams->repair_enabled) {
    FPType energy;
    if (SimAnneal::validatedSearchEnergy(suggested_gs.config, energy, search_finite_model_))
      suggested_gs.system_energy = energy;
    else {
      const auto repaired =
          SimAnneal::repairConfiguration(suggested_gs.config, true);
      if (repaired.valid)
        suggested_gs =
            ChargeConfigResult(repaired.config, true, repaired.energy);
      suggested_gs.repair_attempted = true;
      suggested_gs.repair_budget_exhausted = repaired.budget_exhausted;
    }
  }
  SimAnneal::storeResults(this, thread_id);
}

void SimAnnealThread::genPopDelta(ublas::vector<int> &dn, bool &changed)
{
  if (sparams->population_probability_cache && sparams->n_dbs >= 64)
    genPopDeltaImpl<true>(dn, changed);
  else
    genPopDeltaImpl<false>(dn, changed);
}

template <bool Cached>
void SimAnnealThread::genPopDeltaImpl(ublas::vector<int> &dn, bool &changed)
{
  const simanneal_rng::PopulationProbability<Cached> probability(
      kT, sparams->probability_shortcuts);
  // DB- and DB+ sites can be flipped to DB0, DB0 sites can be flipped to either
  // DB- or DB+ depending on which one is enegetically closer.
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
    if (sparams->transient_domain_mask &&
        !(sparams->final_domains[i] & (1 << (n[i] + change_dir + 1)))) {
      dn[i] = 0;
      continue;
    }
    const FPType draw = uniformDraw();
    const bool accept = probability.accept(draw, x);
    if (accept) {
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

bool SimAnnealThread::acceptHop(const FPType &delta) {
  if (delta < 0)
    return true;
  const FPType draw = uniformDraw();
  return simanneal_rng::hopAcceptance(draw, delta, kT,
                                      sparams->probability_shortcuts);
}

bool SimAnnealThread::evalProb(const FPType &prob)
{
  return prob >= uniformDraw();
}

FPType SimAnnealThread::uniformDraw() {
  return use_pcg ? simanneal_rng::UniformGrid32()(engines.pcg)
                 : dis01(engines.mt);
}

int SimAnnealThread::randInt(const int &min, const int &max) {
  if (max < min)
    throw std::invalid_argument("Invalid inclusive random range");
  if (!use_pcg)
    return RandIntDist(min, max)(engines.mt);
  const std::uint64_t width =
      static_cast<std::uint64_t>(static_cast<std::int64_t>(max) - min) + 1;
  const std::uint32_t draw =
      width == (UINT64_C(1) << 32)
          ? engines.pcg()
          : simanneal_rng::bounded(engines.pcg,
                                   static_cast<std::uint32_t>(width));
  return static_cast<int>(static_cast<std::int64_t>(min) + draw);
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
