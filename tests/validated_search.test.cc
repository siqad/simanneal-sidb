#include "src/charge_domains.h"
#include "src/simanneal.h"
#include "tests/catch2_wrapper.hpp"
#include <limits>

namespace {
phys::SimParams search_fixture() {
  phys::SimParams sp;
  sp.setDBLocs(std::vector<phys::EuclCoord>{
      {0, 0}, {3.84, 0}, {7.68, 0}, {0, 7.68}, {7.68, 7.68}});
  sp.v_ext.clear();
  sp.v_fc.clear();
  sp.mu = -.32;
  sp.anneal_cycles = 32;
  sp.num_instances = 4;
  sp.num_workers = 1;
  sp.deterministic_seed = true;
  sp.random_seed = 731;
  sp.population_backend = phys::PopulationBackend::Portable;
  return sp;
}
bool independent_valid(const phys::SimParams &sp,
                       const phys::ublas::vector<int> &n) {
  const double eps = constants::RECALC_STABILITY_ERR;
  std::vector<double> v(n.size());
  for (unsigned i = 0; i < n.size(); ++i) {
    v[i] = -sp.v_ext[i] - sp.v_fc[i];
    for (unsigned j = 0; j < n.size(); ++j)
      v[i] -= sp.v_ij(i, j) * n[j];
    const double x = v[i] + sp.mu;
    if (!((n[i] == -1 && x < eps) || (n[i] == 1 && x - constants::eta > -eps) ||
          (n[i] == 0 && x > -eps && x - constants::eta < eps)))
      return false;
  }
  for (unsigned i = 0; i < n.size(); ++i)
    for (unsigned j = 0; j < n.size(); ++j)
      if (n[i] < n[j] && -v[i] + v[j] - sp.v_ij(i, j) < -eps)
        return false;
  return true;
}
} // namespace

TEST_CASE("Common validator retains every physical ternary state and domain") {
  for (bool fields : {false, true}) {
    auto sp = search_fixture();
    if (fields) {
      sp.v_ext[0] = .65;
      sp.v_ext[1] = -.65;
      sp.v_fc[2] = -.08;
    }
    phys::SimAnneal model(sp);
    const auto &effective = model.effectiveParams();
    int valid = 0;
    for (int code = 0; code < 243; ++code) {
      phys::ublas::vector<int> q(5);
      int value = code;
      for (int i = 0; i < 5; ++i) {
        q[i] = value % 3 - 1;
        value /= 3;
      }
      const bool expected = independent_valid(effective, q);
      FPType energy = std::numeric_limits<FPType>::quiet_NaN();
      REQUIRE(phys::SimAnneal::isMetastable(q) == expected);
      REQUIRE(phys::SimAnneal::validatedEnergy(q, energy) == expected);
      if (expected) {
        ++valid;
        REQUIRE(energy ==
                Approx(phys::SimAnneal::systemEnergy(q)).margin(1e-12));
        for (int i = 0; i < 5; ++i)
          REQUIRE((effective.final_domains[i] & (1 << (q[i] + 1))) != 0);
      }
    }
    REQUIRE(valid > 0);
    phys::ublas::vector<int> short_config(1);
    FPType ignored;
    REQUIRE_FALSE(phys::SimAnneal::validatedEnergy(short_config, ignored));
  }
}

TEST_CASE(
    "Ordered neutral positive hop has the original Hamiltonian direction") {
  auto sp = search_fixture();
  phys::SimAnneal model(sp);
  phys::ublas::vector<int> q(5);
  q.clear();
  q[1] = 1;
  const auto &physical = model.effectiveParams();
  double vi = -physical.v_ext[0] - physical.v_fc[0],
         vj = -physical.v_ext[1] - physical.v_fc[1];
  for (int k = 0; k < 5; ++k) {
    vi -= physical.v_ij(0, k) * q[k];
    vj -= physical.v_ij(1, k) * q[k];
  }
  const double before = phys::SimAnneal::systemEnergy(q);
  q[0] += 1;
  q[1] -= 1;
  REQUIRE(phys::SimAnneal::systemEnergy(q) - before ==
          Approx(-vi + vj - physical.v_ij(0, 1)).margin(1e-12));
}

TEST_CASE("Model lifetime excludes overlapping models and releases failed "
          "construction") {
  auto sp = search_fixture();
  {
    phys::SimAnneal model(sp);
    REQUIRE_THROWS_AS(phys::SimAnneal(sp), std::logic_error);
    bool blocked = false;
    std::thread competing([&] {
      try {
        phys::SimAnneal other(sp);
      } catch (const std::logic_error &) {
        blocked = true;
      }
    });
    competing.join();
    REQUIRE(blocked);
  }
  auto bad = sp;
  bad.search_profile = static_cast<phys::SearchProfile>(99);
  REQUIRE_THROWS_AS(phys::SimAnneal(bad), std::invalid_argument);
  {
    phys::SimAnneal model(sp);
    REQUIRE(model.effectiveParams().random_backend == phys::RandomBackend::MT);
  }
}

TEST_CASE(
    "Singleton bypass performs no restart and fully validates its result") {
  auto sp = search_fixture();
  sp.setDBLocs(std::vector<phys::EuclCoord>{{0, 0}});
  sp.v_ext[0] = 1;
  sp.v_fc[0] = 0;
  sp.search_profile = phys::SearchProfile::Optimized;
  phys::SimAnneal model(sp);
  model.invokeSimAnneal();
  REQUIRE(model.searchStats().singleton_used);
  REQUIRE(model.searchStats().executed_restarts == 0);
  REQUIRE(model.suggestedResults().size() == 1);
  REQUIRE(model.suggestedResults()[0].config[0] == -1);
  REQUIRE(phys::SimAnneal::isMetastable(model.suggestedResults()[0].config));
}

TEST_CASE("Bounded repair returns original fully checked states") {
  auto sp = search_fixture();
  sp.v_ext[0] = .65;
  sp.v_ext[1] = -.65;
  phys::SimAnneal model(sp);
  phys::ublas::vector<int> q(5);
  q.clear();
  const auto result = phys::SimAnneal::repairConfiguration(q);
  REQUIRE(result.hops <= 20);
  REQUIRE(result.repair_passes <= 8);
  REQUIRE(result.valid);
  REQUIRE(phys::SimAnneal::isMetastable(result.config));
  REQUIRE(result.energy ==
          Approx(phys::SimAnneal::systemEnergy(result.config)).margin(1e-12));
}

TEST_CASE(
    "Finite grid shortcuts preserve probability decisions and zero endpoints") {
  const double kT = .01;
  for (double draw :
       {0., (1.0 / 4294967296.0), .25, .5, 1. - (1.0 / 4294967296.0)})
    for (double x : {-100., -.401, -.4, -.399, 0., .399, .4, .401, 100.}) {
      REQUIRE(simanneal_rng::populationAcceptance(draw, x, kT, true) ==
              simanneal_rng::populationAcceptance(draw, x, kT, false));
      if (x >= 0)
        REQUIRE(simanneal_rng::hopAcceptance(draw, x, kT, true) ==
                simanneal_rng::hopAcceptance(draw, x, kT, false));
    }
  REQUIRE(simanneal_rng::populationAcceptance(0, 1000, kT, true));
  REQUIRE(simanneal_rng::hopAcceptance(0, 1000, kT, true));
  simanneal_rng::Pcg32 pcg;
  pcg.reseed(42, 54);
  REQUIRE(pcg() == 0xa15c02b7U);
  REQUIRE(pcg() == 0x7b47f409U);
  REQUIRE_THROWS_AS(simanneal_rng::bounded(pcg, 0), std::invalid_argument);
  for (int i = 0; i < 1000; ++i)
    REQUIRE(simanneal_rng::bounded(pcg, 7) < 7);
}

TEST_CASE("PCG optimized scheduling is independent of worker count") {
  auto sp = search_fixture();
  sp.search_profile = phys::SearchProfile::Optimized;
  sp.singleton_shortcut = phys::FeatureSetting::Disabled;
  phys::SuggestedResults expected;
  for (int workers : {1, 4}) {
    sp.num_workers = workers;
    phys::SimAnneal model(sp);
    model.invokeSimAnneal();
    REQUIRE(model.searchStats().executed_restarts == 4);
    if (expected.empty())
      expected = model.suggestedResults();
    REQUIRE(expected.size() == model.suggestedResults().size());
    for (unsigned i = 0; i < expected.size(); ++i) {
      REQUIRE(phys::SimAnneal::configToStr(expected[i].config) ==
              phys::SimAnneal::configToStr(model.suggestedResults()[i].config));
      REQUIRE(expected[i].system_energy ==
              model.suggestedResults()[i].system_energy);
    }
  }
}

TEST_CASE("Fixed-charge validation retains prior fields on malformed input") {
  auto sp = search_fixture();
  sp.v_fc[0] = .125;
  const std::vector<phys::EuclCoord3d> location{{1, 2, 3}};
  REQUIRE_THROWS_AS(sp.setFixedCharges(location, {}, {5.6}, {5}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(sp.setFixedCharges(location, {1}, {0}, {5}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(sp.setFixedCharges(location, {1}, {5.6}, {0}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(sp.setFixedCharges({{0, 0, 0}}, {1}, {5.6}, {5}),
                    std::invalid_argument);
  REQUIRE(sp.v_fc[0] == .125);
  sp.setFixedCharges(location, {-1}, {5.6}, {5});
  REQUIRE(std::isfinite(sp.v_fc[0]));
  REQUIRE(sp.v_fc[0] < 0);
}

TEST_CASE("Profile defaults and explicit feature overrides resolve in solver") {
  auto sp = search_fixture();
  {
    phys::SimAnneal model(sp);
    REQUIRE_FALSE(model.effectiveParams().repair_enabled);
    REQUIRE_FALSE(model.effectiveParams().singleton_enabled);
    REQUIRE(model.effectiveParams().random_backend == phys::RandomBackend::MT);
  }
  sp.search_profile = phys::SearchProfile::Optimized;
  {
    phys::SimAnneal model(sp);
    REQUIRE(model.effectiveParams().repair_enabled);
    REQUIRE(model.effectiveParams().singleton_enabled);
    REQUIRE(model.effectiveParams().random_backend ==
            phys::RandomBackend::PCG32);
  }
  sp.repair = phys::FeatureSetting::Disabled;
  sp.singleton_shortcut = phys::FeatureSetting::Disabled;
  sp.random_backend = phys::RandomBackend::MT;
  {
    phys::SimAnneal model(sp);
    REQUIRE_FALSE(model.effectiveParams().repair_enabled);
    REQUIRE_FALSE(model.effectiveParams().singleton_enabled);
    REQUIRE(model.effectiveParams().random_backend == phys::RandomBackend::MT);
  }
}

TEST_CASE(
    "Final validation rejects newly strong hops after public model mutation") {
  auto sp = search_fixture();
  std::vector<phys::EuclCoord> points;
  for (int i = 0; i < 256; ++i)
    points.emplace_back((i % 16) * 50., (i / 16) * 50.);
  sp.setDBLocs(points);
  sp.v_ext.clear();
  sp.v_fc.clear();
  sp.repair = phys::FeatureSetting::Enabled;
  phys::SimAnneal model(sp);
  auto &physical = phys::SimAnneal::sim_params;
  phys::ublas::vector<int> q(256);
  for (int i = 0; i < 256; ++i)
    q[i] = -1;
  q[0] = 0;
  q[1] = 1;
  const auto restore_potentials = [&] {
    for (int i = 0; i < 256; ++i) {
      const double target = -physical.mu + (q[i] == -1  ? -.1
                                            : q[i] == 0 ? constants::eta * .5
                                                        : constants::eta + .1);
      double interactions = 0;
      for (int j = 0; j < 256; ++j)
        interactions += physical.v_ij(i, j) * q[j];
      physical.v_ext[i] = -target - interactions;
    }
  };
  REQUIRE(physical.v_ij(0, 1) <= .05);
  restore_potentials();
  REQUIRE(independent_valid(physical, q));
  REQUIRE(phys::SimAnneal::isMetastable(q));
  physical.v_ij(0, 1) = physical.v_ij(1, 0) = .6;
  restore_potentials();
  REQUIRE_FALSE(independent_valid(physical, q));
  REQUIRE_FALSE(phys::SimAnneal::isMetastable(q));
  FPType energy;
  REQUIRE_FALSE(phys::SimAnneal::validatedEnergy(q, energy));
}

TEST_CASE("Fused validated energy rejects a nonzero interaction diagonal") {
  auto sp = search_fixture();
  sp.setDBLocs(std::vector<phys::EuclCoord>{{0, 0}});
  sp.v_ext[0] = 1;
  sp.v_fc[0] = 0;
  phys::SimAnneal model(sp);
  phys::ublas::vector<int> q(1);
  q[0] = -1;
  FPType energy;
  REQUIRE(phys::SimAnneal::validatedEnergy(q, energy));
  phys::SimAnneal::sim_params.v_ij(0, 0) = .5;
  REQUIRE_FALSE(phys::SimAnneal::validatedEnergy(q, energy));
}

TEST_CASE("Common validator preserves legacy population boundary association") {
  auto sp = search_fixture();
  sp.setDBLocs(std::vector<phys::EuclCoord>{{0, 0}});
  sp.v_ext[0] = 0;
  sp.v_fc[0] = 0;
  phys::SimAnneal model(sp);
  auto &physical = phys::SimAnneal::sim_params;
  phys::ublas::vector<int> q(1);
  const double eps = constants::RECALC_STABILITY_ERR;
  for (int charge : {-1, 0, 1})
    for (double sign : {-1., 1.}) {
      const double center =
          charge == -1 ? -physical.mu : constants::eta - physical.mu;
      const double boundary = center + sign * eps;
      for (double v :
           {std::nextafter(boundary, -std::numeric_limits<double>::infinity()),
            boundary,
            std::nextafter(boundary,
                           std::numeric_limits<double>::infinity())}) {
        physical.v_ext[0] = -v;
        q[0] = charge;
        const double lower = v + physical.mu,
                     upper = v + (physical.mu - constants::eta);
        const bool expected = (charge == -1 && lower < eps) ||
                              (charge == 1 && upper > -eps) ||
                              (charge == 0 && lower > -eps && upper < eps);
        FPType energy;
        REQUIRE(phys::SimAnneal::isMetastable(q) == expected);
        REQUIRE(phys::SimAnneal::validatedEnergy(q, energy) == expected);
      }
    }
}

TEST_CASE("Common validator retains representable extreme field energy") {
  auto sp = search_fixture();
  sp.setDBLocs(std::vector<phys::EuclCoord>{{0, 0}});
  sp.v_ext[0] = 0;
  sp.v_fc[0] = 0;
  phys::SimAnneal model(sp);
  const double field = std::numeric_limits<double>::max() * .75;
  phys::SimAnneal::sim_params.v_ext[0] = field;
  phys::ublas::vector<int> q(1);
  q[0] = -1;
  FPType energy;
  REQUIRE(phys::SimAnneal::isMetastable(q));
  REQUIRE(phys::SimAnneal::validatedEnergy(q, energy));
  REQUIRE(energy == -field);
  REQUIRE(energy == phys::SimAnneal::systemEnergy(q));
}

TEST_CASE("Model ownership can be released on another thread") {
  auto sp = search_fixture();
  auto *model = new phys::SimAnneal(sp);
  std::thread destroyer([&] { delete model; });
  destroyer.join();
  phys::SimAnneal replacement(sp);
  REQUIRE(replacement.effectiveParams().n_dbs == sp.n_dbs);
}
