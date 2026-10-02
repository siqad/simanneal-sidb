// Single-worker benchmark. Uses the production annealing loop and validator.
#include "src/simanneal.h"
#include <boost/property_tree/json_parser.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace phys;
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b-a).count();
}
int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: anneal_bench requests.json");
    boost::property_tree::ptree root;
    boost::property_tree::read_json(argv[1], root);
    std::cout << std::setprecision(17);
    for (const auto &request : root.get_child("requests")) {
      const auto &r = request.second;
      SimParams sp;
      std::vector<EuclCoord> points;
      for (const auto &point : r.get_child("points")) {
        auto it = point.second.begin();
        const double x = (it++)->second.get_value<double>();
        points.emplace_back(x, it->second.get_value<double>());
      }
      sp.setDBLocs(points);
      sp.v_ext.clear(); sp.v_fc.clear();
      sp.num_instances = 1;
      sp.result_queue_factor = 0; // same one-entry history for every policy
      const std::string backend = r.get<std::string>("backend", "auto");
      if (backend == "portable") sp.population_backend = PopulationBackend::Portable;
      else if (backend == "accelerate") sp.population_backend = PopulationBackend::Accelerate;
      else if (backend != "auto") throw std::invalid_argument("unknown benchmark backend");
      sp.mu = r.get<double>("mu", -0.25);
      sp.anneal_cycles = r.get<int>("cycles", 512);
      sp.hop_attempt_factor = r.get<int>("hop_factor", 5);
      sp.T_init = r.get<double>("temperature", 500);
      sp.T_e_inv_point = r.get<double>("cooling", 0.09995);
      sp.v_freeze_end_point = r.get<double>("freeze", 0.4);
      sp.strategic_v_freeze_reset = r.get<bool>("reset", false);
      sp.hop_selection = static_cast<HopSelection>(r.get<int>("policy", 0));
      sp.hop_neighbors = r.get<int>("neighbors", 16);
      sp.hop_length_nm = r.get<double>("length_nm", 2);
      sp.hop_global_probability = r.get<double>("global", 0.2);
      const auto start = Clock::now();
      SimAnneal master(sp);
      const auto initialized = Clock::now();
      // Avoid OS scheduling noise: one production worker, in the calling thread.
      SimAnnealThread worker(0, r.get<std::uint64_t>("seed"));
      worker.run();
      const auto annealed = Clock::now();
      const auto results = master.suggestedConfigResults(true);
      const auto finished = Clock::now();
      std::cout << "{\"id\":\"" << r.get<std::string>("id") << "\",\"init_ms\":" << ms(start, initialized)
        << ",\"anneal_ms\":" << ms(initialized, annealed)
        << ",\"validate_ms\":" << ms(annealed, finished)
        << ",\"total_ms\":" << ms(start, finished)
        << ",\"neighbor_bytes\":" << SimAnneal::sim_params.hop_neighborhood.storageBytes()
        << ",\"valid\":" << (!results.empty() ? "true" : "false");
      if (!results.empty()) {
        std::cout << ",\"energy\":" << results[0].system_energy << ",\"config\":[";
        for (std::size_t i=0; i<results[0].config.size(); ++i) {
          if (i) std::cout << ',';
          std::cout << results[0].config[i];
        }
        std::cout << ']';
      }
      // Evaluate references with the very same Hamiltonian and full validator.
      const auto references = r.get_child_optional("references");
      if (references) {
        std::cout << ",\"references\":[";
        bool first = true;
        for (const auto &ref : *references) {
          ublas::vector<int> charge(points.size());
          std::size_t j = 0;
          for (const auto &value : ref.second) charge[j++] = value.second.get_value<int>();
          if (j != points.size()) throw std::invalid_argument("reference size mismatch");
          if (!first) std::cout << ',';
          first = false;
          std::cout << "{\"valid\":" << (SimAnneal::isMetastable(charge) ? "true" : "false")
            << ",\"energy\":" << SimAnneal::systemEnergy(charge) << '}';
        }
        std::cout << ']';
      }
      std::cout << "}\n" << std::flush;
    }
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
