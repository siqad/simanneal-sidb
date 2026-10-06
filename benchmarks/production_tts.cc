// Integrated production compute benchmark; no research-only postprocessing.
// Copyright 2026 SiQAD contributors. Licensed under Apache-2.0.
#include "src/population_blas.h"
#include "src/simanneal.h"
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Tree = boost::property_tree::ptree;
using Clock = std::chrono::steady_clock;

struct Observation {
  bool valid = false;
  double energy = 0;
  std::vector<int> config;
  double setup_ms = 0, invoke_ms = 0, export_ms = 0, cleanup_ms = 0,
         total_ms = 0;
  int restarts = 0, workers = 0, cycles = 0, hop_factor = 0;
  bool repair = false, singleton = false, probability = false,
       transient = false, population_probability_cache = false;
  int rng = 0, backend = 0, refinement = 0, candidates = 0, rounds = 0;
  int blas_threads = 0;
  phys::SearchStats stats;
};

double milliseconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

std::string quote(const std::string &value) {
  std::ostringstream output;
  output << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\')
      output << '\\' << c;
    else if (c < 32) {
      output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
             << static_cast<unsigned>(c) << std::dec;
    } else
      output << c;
  }
  output << '"';
  return output.str();
}

void fillPotential(const Tree &input, const char *key,
                   phys::ublas::vector<FPType> &potential) {
  const auto &values = input.get_child(key);
  if (values.size() != potential.size())
    throw std::invalid_argument(std::string(key) + " has incorrect dimensions");
  std::size_t index = 0;
  for (const auto &entry : values)
    potential[index++] = entry.second.get_value<double>();
}

Observation execute(const Tree &request) {
  Observation observation;
  const auto started = Clock::now();
  Clock::time_point initialized, invoked, exported;
  {
    // Coordinate vectors, SimParams matrices, solver caches, workers, result
    // validation and solver-owned cleanup are all inside this compute interval.
    std::vector<phys::EuclCoord> coordinates;
    const auto &points = request.get_child("points");
    coordinates.reserve(points.size());
    for (const auto &point : points) {
      if (point.second.size() != 2)
        throw std::invalid_argument("Each point must have two coordinates");
      auto component = point.second.begin();
      const double x = component++->second.get_value<double>();
      const double y = component->second.get_value<double>();
      coordinates.emplace_back(x, y);
    }
    phys::SimParams params;
    params.setDBLocs(coordinates);
    params.num_instances = request.get<int>("restarts");
    params.num_workers = request.get<int>("workers");
    params.anneal_cycles = request.get<int>("cycles");
    params.hop_attempt_factor = request.get<int>("hop_factor");
    params.T_e_inv_point = request.get<double>("cooling");
    params.v_freeze_end_point = request.get<double>("freeze");
    params.T_init = request.get<double>("temperature");
    params.T_min = request.get<double>("T_min");
    params.mu = request.get<double>("mu");
    params.eps_r = request.get<double>("epsilon_r");
    params.debye_length = request.get<double>("lambda_tf");
    params.deterministic_seed = true;
    params.random_seed = request.get<std::uint64_t>("seed");
    params.result_queue_factor = 0;
    params.record_history = false;
    fillPotential(request, "external_potential", params.v_ext);
    fillPotential(request, "fixed_potential", params.v_fc);

    const auto profile = request.get<std::string>("search_profile");
    if (profile == "legacy")
      params.search_profile = phys::SearchProfile::Legacy;
    else if (profile == "optimized")
      params.search_profile = phys::SearchProfile::Optimized;
    else
      throw std::invalid_argument("Unknown search_profile");
    const auto backend = request.get<std::string>("backend");
    if (backend == "portable")
      params.population_backend = phys::PopulationBackend::Portable;
    else if (backend == "openblas")
      params.population_backend = phys::PopulationBackend::OpenBLAS;
    else
      throw std::invalid_argument("Unknown backend");
    params.probability_shortcuts = request.get<bool>("probability_shortcuts");
    params.population_probability_cache =
        request.get<bool>("population_probability_cache", false);
    params.transient_domain_mask = request.get<bool>("transient_domain_mask");
    const auto mode = request.get<std::string>("refinement");
    auto &options = params.refinement_options;
    if (mode == "none")
      options.mode = phys::refinement::Mode::Disabled;
    else if (mode == "k6")
      options.mode = phys::refinement::Mode::K6;
    else if (mode == "k10")
      options.mode = phys::refinement::Mode::K10;
    else if (mode == "shared")
      options.mode = phys::refinement::Mode::SharedK10;
    else
      throw std::invalid_argument("Unknown refinement");
    options.candidates = request.get<int>("refinement_candidates");
    options.rounds = request.get<int>("refinement_rounds");
    options.trials = 1;

    phys::SimAnneal solver(params);
    const auto &effective = solver.effectiveParams();
    observation.restarts = effective.num_instances;
    observation.workers = effective.num_workers;
    observation.cycles = effective.anneal_cycles;
    observation.hop_factor = effective.hop_attempt_factor;
    observation.repair = effective.repair_enabled;
    observation.singleton = effective.singleton_enabled;
    observation.probability = effective.probability_shortcuts;
    observation.population_probability_cache = effective.population_probability_cache;
    observation.transient = effective.transient_domain_mask;
    observation.rng = static_cast<int>(effective.random_backend);
    observation.backend = static_cast<int>(effective.population_backend);
    observation.refinement =
        static_cast<int>(effective.refinement_options.mode);
    observation.candidates = effective.refinement_options.candidates;
    observation.rounds = effective.refinement_options.rounds;
    initialized = Clock::now();
    solver.invokeSimAnneal();
    invoked = Clock::now();
    auto results = solver.suggestedConfigResults(true);
    // Production export already independently validates these records. The
    // harness only reduces to the lowest valid exported energy.
    for (const auto &result : results) {
      if (!std::isfinite(result.system_energy))
        continue;
      if (!observation.valid || result.system_energy < observation.energy) {
        observation.valid = true;
        observation.energy = result.system_energy;
        observation.config.assign(result.config.begin(), result.config.end());
      }
    }
    observation.stats = solver.searchStats();
    observation.blas_threads = observation.stats.population_blas_threads;
    exported = Clock::now();
  }
  const auto finished = Clock::now();
  observation.setup_ms = milliseconds(started, initialized);
  observation.invoke_ms = milliseconds(initialized, invoked);
  observation.export_ms = milliseconds(invoked, exported);
  observation.cleanup_ms = milliseconds(exported, finished);
  observation.total_ms = milliseconds(started, finished);
  return observation;
}

void write(const Tree &request, const Observation &row) {
  std::cout << std::setprecision(17) << std::boolalpha
            << "{\"id\":" << quote(request.get<std::string>("id"))
            << ",\"valid\":" << row.valid << ",\"energy\":";
  if (row.valid)
    std::cout << row.energy;
  else
    std::cout << "null";
  std::cout << ",\"config\":[";
  for (std::size_t i = 0; i < row.config.size(); ++i) {
    if (i)
      std::cout << ',';
    std::cout << row.config[i];
  }
  std::cout << "],\"total_ms\":" << row.total_ms
            << ",\"setup_ms\":" << row.setup_ms
            << ",\"invoke_ms\":" << row.invoke_ms
            << ",\"export_ms\":" << row.export_ms
            << ",\"cleanup_ms\":" << row.cleanup_ms
            << ",\"restarts\":" << row.restarts
            << ",\"workers\":" << row.workers << ",\"cycles\":" << row.cycles
            << ",\"hop_factor\":" << row.hop_factor
            << ",\"repair_enabled\":" << row.repair
            << ",\"singleton_enabled\":" << row.singleton
            << ",\"probability_shortcuts\":" << row.probability
            << ",\"population_probability_cache\":" << row.population_probability_cache
            << ",\"transient_domain_mask\":" << row.transient
            << ",\"random_backend_enum\":" << row.rng
            << ",\"population_backend_enum\":" << row.backend
            << ",\"refinement_enum\":" << row.refinement
            << ",\"blas_threads\":" << row.blas_threads
            << ",\"refinement_candidates\":" << row.candidates
            << ",\"refinement_rounds\":" << row.rounds
            << ",\"executed_restarts\":" << row.stats.executed_restarts
            << ",\"repair_attempts\":" << row.stats.repair_attempts
            << ",\"repair_budget_exhaustions\":"
            << row.stats.repair_budget_exhaustions
            << ",\"singleton_used\":" << row.stats.singleton_used
            << ",\"refinement_patterns\":" << row.stats.refinement.patterns
            << ",\"refinement_selected\":" << row.stats.refinement.selected
            << ",\"refinement_trials\":" << row.stats.refinement.trials
            << ",\"refinement_improvements\":"
            << row.stats.refinement.improvements
            << ",\"refinement_budget_exhausted\":"
            << row.stats.refinement.budget_exhausted
            << ",\"refinement_geometry_count\":"
            << row.stats.refinement.geometry_count
            << ",\"refinement_shared_single_cache_fallback\":"
            << row.stats.refinement.shared_single_cache_fallback
            << ",\"refinement_center_offset\":"
            << row.stats.refinement_center_offset
            << ",\"refinement_geometry_bytes\":"
            << row.stats.refinement.geometry_bytes
            << ",\"refinement_dedup_bytes\":"
            << row.stats.refinement.dedup_bytes << "}\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("Usage: production_tts REQUESTS.json");
    Tree input;
    boost::property_tree::read_json(argv[1], input);
    for (const auto &request : input.get_child("requests")) {
      const auto row = execute(request.second);
      write(request.second, row);
    }
  } catch (const std::exception &error) {
    std::cerr << "production_tts: " << error.what() << '\n';
    return 1;
  }
}
