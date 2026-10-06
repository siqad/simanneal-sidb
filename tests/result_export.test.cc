#include "src/simanneal.h"
#include "tests/catch2_wrapper.hpp"
#include <cstring>

namespace phys {
// This friend has no production definition or public solver accessor.
struct SimAnnealExportTestAccess {
  static SuggestedResults &results() { return SimAnneal::suggested_gs_results; }
  static std::size_t workers(const SimAnneal &solver) { return solver.export_workers_; }
  static std::size_t launches(const SimAnneal &solver) { return solver.export_async_launches_; }
};
}
namespace {
bool energy_bits_equal(FPType left, FPType right) {
  return std::memcmp(&left, &right, sizeof(FPType)) == 0;
}

void require_same_results(const phys::SuggestedResults &serial,
                          const phys::SuggestedResults &parallel) {
  REQUIRE(parallel.size() == serial.size());
  for (std::size_t i = 0; i < serial.size(); ++i) {
    REQUIRE(parallel[i].config.size() == serial[i].config.size());
    for (std::size_t j = 0; j < serial[i].config.size(); ++j)
      REQUIRE(parallel[i].config[j] == serial[i].config[j]);
    REQUIRE(energy_bits_equal(parallel[i].system_energy,
                              serial[i].system_energy));
    REQUIRE(parallel[i].initialized == serial[i].initialized);
    REQUIRE(parallel[i].pop_likely_stable == serial[i].pop_likely_stable);
    REQUIRE(parallel[i].refinement_result == serial[i].refinement_result);
    REQUIRE(parallel[i].repair_attempted == serial[i].repair_attempted);
    REQUIRE(parallel[i].repair_budget_exhausted == serial[i].repair_budget_exhausted);
  }
}

void require_config_code(const phys::ChargeConfigResult &result, unsigned code) {
  REQUIRE(result.config.size() == 404);
  for (unsigned site = 0; site < 404; ++site) {
    const int expected = site < 5 && ((code >> site) & 1) ? -1 : 0;
    REQUIRE(result.config[site] == expected);
  }
}
} // namespace

TEST_CASE("Parallel tidy export validates the current model and preserves order",
          "[export]") {
  phys::SimParams params;
  std::vector<phys::EuclCoord> points;
  for (unsigned i = 0; i < 404; ++i)
    points.emplace_back(i * 3.84, 0.0);
  params.setDBLocs(points);
  params.v_ext.clear();
  params.v_fc.clear();
  params.num_instances = 4;
  params.num_workers = 4;
  params.population_backend = phys::PopulationBackend::Portable;
  phys::SimAnneal solver(params);

  // Exercise the public mutable model contract between synchronous exports.
  // The degenerate model makes all {-1,0} configurations physically stable.
  auto &model = phys::SimAnneal::sim_params;
  model.v_ij.clear();
  model.v_ext.clear();
  model.v_fc.clear();
  model.mu = 0;

  // The underlying object is nonconst. Injection is confined to this test.
  auto &raw = phys::SimAnnealExportTestAccess::results();
  raw.clear();
  for (unsigned code = 0; code < 32; ++code) {
    phys::ublas::vector<int> configuration(404);
    configuration.clear();
    for (unsigned site = 0; site < 5; ++site)
      if ((code >> site) & 1)
        configuration[site] = -1;
    // Incorrect public metadata must not bypass full validation or energy work.
    raw.emplace_back(configuration, false, FPType(9000 + code));
    raw.back().refinement_result = code % 2;
    raw.back().repair_attempted = code % 3;
    raw.back().repair_budget_exhausted = code % 5;
  }
  raw.push_back(raw[7]); // Duplicate retains the first occurrence only.
  raw.back().pop_likely_stable = true;
  raw.back().system_energy = -12345;
  phys::ChargeConfigResult uninitialized = raw[3];
  uninitialized.initialized = false;
  // Give this record a unique, otherwise valid configuration, so dedup cannot
  // conceal an incorrect failure to filter its initialized flag.
  uninitialized.config[6] = -1;
  raw.push_back(uninitialized);
  phys::ublas::vector<int> invalid(404);
  invalid.clear();
  invalid[0] = 1;
  raw.emplace_back(invalid, true, FPType(-9999));

  phys::SuggestedResults expected_untidy;
  for (const auto &result : raw)
    if (result.initialized)
      expected_untidy.push_back(result);
  require_same_results(expected_untidy, solver.suggestedConfigResults(false));
  model.num_workers = 1;
  const auto serial = solver.suggestedConfigResults(true);
  REQUIRE(phys::SimAnnealExportTestAccess::workers(solver) == 1);
  REQUIRE(phys::SimAnnealExportTestAccess::launches(solver) == 0);
  model.num_workers = 4;
  const auto parallel = solver.suggestedConfigResults(true);
  REQUIRE(phys::SimAnnealExportTestAccess::workers(solver) == 4);
  REQUIRE(phys::SimAnnealExportTestAccess::launches(solver) == 4);
  REQUIRE(serial.size() == 32);
  require_same_results(serial, parallel);
  for (unsigned code = 0; code < 32; ++code) {
    require_config_code(parallel[code], code);
    REQUIRE(parallel[code].system_energy == FPType(0));
    REQUIRE_FALSE(parallel[code].pop_likely_stable);
    REQUIRE(parallel[code].refinement_result == bool(code % 2));
    REQUIRE(parallel[code].repair_attempted == bool(code % 3));
    REQUIRE(parallel[code].repair_budget_exhausted == bool(code % 5));
  }

  // A new field rejects every configuration neutral at site zero. Export must
  // validate again and recalculate the surviving configurations' energy.
  model.v_ext[0] = 1;
  model.num_workers = 1;
  const auto serial_changed = solver.suggestedConfigResults(true);
  REQUIRE(phys::SimAnnealExportTestAccess::workers(solver) == 1);
  REQUIRE(phys::SimAnnealExportTestAccess::launches(solver) == 0);
  model.num_workers = 4;
  const auto parallel_changed = solver.suggestedConfigResults(true);
  REQUIRE(phys::SimAnnealExportTestAccess::workers(solver) == 4);
  REQUIRE(phys::SimAnnealExportTestAccess::launches(solver) == 4);
  REQUIRE(serial_changed.size() == 16);
  require_same_results(serial_changed, parallel_changed);
  for (unsigned i = 0; i < 16; ++i) {
    require_config_code(parallel_changed[i], 2 * i + 1);
    REQUIRE(parallel_changed[i].system_energy == FPType(-1));
  }

  // Untidy export retains the stored energies, metadata, invalid state,
  // duplicate, and order even after the public model changes.
  require_same_results(expected_untidy, solver.suggestedConfigResults(false));
}

TEST_CASE("Tidy export bounds worker count and uses all parallel thresholds", "[export]") {
  struct Case { int sites, unique, configured, workers; };
  for (const auto &test : {Case{512,31,8,1}, Case{512,32,8,4},
                          Case{127,256,4,1}, Case{128,244,4,1},
                          Case{128,245,4,4}, Case{512,32,2,2}}) {
    CAPTURE(test.sites, test.unique, test.configured);
    phys::SimParams params;
    std::vector<phys::EuclCoord> points;
    for (int i = 0; i < test.sites; ++i)
      points.emplace_back(i * 3.84, 0.);
    params.setDBLocs(points);
    params.num_instances = test.configured;
    params.num_workers = test.configured;
    params.population_backend = phys::PopulationBackend::Portable;
    phys::SimAnneal solver(params);
    auto &model = phys::SimAnneal::sim_params;
    model.v_ij.clear(); model.v_ext.clear(); model.v_fc.clear(); model.mu = 0;
    auto &raw = phys::SimAnnealExportTestAccess::results();
    raw.clear();
    for (int code = 0; code < test.unique; ++code) {
      phys::ublas::vector<int> q(test.sites);
      q.clear();
      for (int bit = 0; bit < 8; ++bit)
        q[bit] = (code >> bit) & 1 ? -1 : 0;
      raw.emplace_back(q, false, 123.);
    }
    // Many duplicates must not trigger parallel validation by themselves.
    for (int i = 0; i < 256; ++i)
      raw.push_back(raw.front());
    const auto output = solver.suggestedConfigResults(true);
    REQUIRE(output.size() == std::size_t(test.unique));
    REQUIRE(phys::SimAnnealExportTestAccess::workers(solver) == std::size_t(test.workers));
    REQUIRE(phys::SimAnnealExportTestAccess::launches(solver) ==
            std::size_t(test.workers == 1 ? 0 : test.workers));
  }
}

TEST_CASE("Empty tidy export and malformed dimensions produce no result", "[export]") {
  phys::SimParams params;
  params.setDBLocs(std::vector<phys::EuclCoord>{{0,0},{3.84,0}});
  params.num_instances = 1; params.num_workers = 1;
  phys::SimAnneal solver(params);
  auto &raw = phys::SimAnnealExportTestAccess::results();
  raw.clear();
  REQUIRE(solver.suggestedConfigResults(true).empty());
  phys::ublas::vector<int> q(1); q[0] = -1;
  raw.emplace_back(q, true, -99.);
  raw.emplace_back(phys::ublas::vector<int>(), true, -99.);
  raw.emplace_back();
  REQUIRE(solver.suggestedConfigResults(true).empty());
  REQUIRE(solver.suggestedConfigResults(false).size() == 2);
}
