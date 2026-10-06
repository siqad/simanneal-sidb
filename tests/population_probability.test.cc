#include "src/simanneal.h"
#include "tests/catch2_wrapper.hpp"
#include <limits>

TEST_CASE("Population cache default retains original scalar acceptance") {
  REQUIRE_FALSE(phys::SimParams().population_probability_cache);
  for (bool shortcut : {false, true})
    for (double kT : {.001, .01, 1., 40.}) {
      simanneal_rng::PopulationProbability<false> original(kT, shortcut);
      simanneal_rng::PopulationProbability<true> cached(kT, shortcut);
      REQUIRE(cached.use_inverse);
      REQUIRE(cached.inverse == 1. / kT);
      for (double draw : {0., 1. / 4294967296., .25, .5, 1. - 1. / 4294967296.})
        for (double x : {-100., -.401, -.4, -.399, 0., .399, .4, .401, 100.}) {
          REQUIRE(original.accept(draw, x) ==
                  simanneal_rng::populationAcceptance(draw, x, kT, shortcut));
          const double bound = shortcut ? 40 * kT : 0;
          const bool expected = bound > 0 && x > bound ? draw == 0
              : bound > 0 && x < -bound ? true
              : draw <= 1. / (1 + std::exp(x * (1. / kT)));
          REQUIRE(cached.accept(draw, x) == expected);
        }
    }
}

TEST_CASE("Population cache falls back to division for exceptional temperatures") {
  const double tiny = std::numeric_limits<double>::denorm_min();
  for (bool shortcut : {false, true})
    for (double kT : {tiny, std::nextafter(std::numeric_limits<double>::min(), 0.),
                      0., -0., -tiny, -.01,
                      std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::quiet_NaN()}) {
      simanneal_rng::PopulationProbability<true> cached(kT, shortcut);
      REQUIRE_FALSE(cached.use_inverse);
      for (double x : {-1., -tiny, -0., 0., tiny, 1.})
        for (double draw : {0., .25, .5, 1.})
          REQUIRE(cached.accept(draw, x) ==
                  simanneal_rng::populationAcceptance(draw, x, kT, shortcut));
    }
  simanneal_rng::PopulationProbability<true> smallest(
      std::numeric_limits<double>::min(), false);
  REQUIRE(smallest.use_inverse);
  REQUIRE(std::isfinite(smallest.inverse));
}

TEST_CASE("Population cache preserves small-layout paths and repeatable large-layout histories") {
  for (auto profile : {phys::SearchProfile::Legacy, phys::SearchProfile::Optimized})
    for (int sites : {3, 64}) {
      phys::AllChargeResults baseline;
      for (bool cache : {false, true}) {
        phys::SimParams params;
        std::vector<phys::EuclCoord> points;
        for (int i = 0; i < sites; ++i)
          points.emplace_back((i % 8) * 7.68, (i / 8) * 7.68);
        params.setDBLocs(points);
        params.v_ext.clear(); params.v_fc.clear();
        params.search_profile = profile;
        params.population_probability_cache = cache;
        params.num_instances = 2; params.num_workers = 1;
        params.anneal_cycles = 32;
        params.deterministic_seed = true; params.random_seed = 731;
        params.record_history = true; params.result_queue_factor = 1;
        phys::SimAnneal solver(params);
        REQUIRE(solver.effectiveParams().population_probability_cache == cache);
        solver.invokeSimAnneal();
        const auto first = solver.chargeResults();
        if (!cache)
          baseline = first;
        solver.invokeSimAnneal();
        for (std::size_t restart = 0; restart < first.size(); ++restart) {
          REQUIRE(first[restart].size() == 32);
          REQUIRE(solver.chargeResults()[restart].size() == first[restart].size());
          for (std::size_t cycle = 0; cycle < first[restart].size(); ++cycle) {
            const auto &before = first[restart][cycle];
            const auto &after = solver.chargeResults()[restart][cycle];
            REQUIRE(phys::SimAnneal::configToStr(before.config) ==
                    phys::SimAnneal::configToStr(after.config));
            REQUIRE(before.system_energy == after.system_energy);
            if (sites < 64) {
              REQUIRE(phys::SimAnneal::configToStr(before.config) ==
                      phys::SimAnneal::configToStr(baseline[restart][cycle].config));
              REQUIRE(before.system_energy == baseline[restart][cycle].system_energy);
            }
          }
        }
      }
    }
}
