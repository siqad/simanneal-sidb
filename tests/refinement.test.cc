#include "src/refinement.h"
#include "tests/catch2_wrapper.hpp"
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
using namespace phys::refinement;
namespace {
struct Fixture {
  Matrix a;
  Potentials external, fixed;
  std::vector<unsigned char> domains;
  Fixture(int pairs)
      : a(2 * pairs, 2 * pairs), external(2 * pairs), fixed(2 * pairs),
        domains(2 * pairs, 3) {
    a.clear();
    external.clear();
    fixed.clear();
    for (int i = 0; i < 2 * pairs; ++i) {
      if (i % 2 == 0) {
        external[i] = .004;
        fixed[i] = .006;
      }
      for (int j = 0; j < i; ++j) {
        const double value = i / 2 == j / 2 ? .5 : i % 2 == j % 2 ? .01 : .06;
        a(i, j) = a(j, i) = value;
      }
    }
  }
  ModelView model() const {
    return ModelView(a, external, fixed, domains, -.25, .59, 1e-6);
  }
  Charges aligned(int parity) const {
    Charges q(a.size1());
    q.clear();
    for (std::size_t i = parity; i < q.size(); i += 2)
      q[i] = -1;
    return q;
  }
};
// Independent full Hamiltonian, all population sites and all ordered hops.
// Unlike the cluster kernel, this evaluator never fixes or omits outsider
// sites.

bool full(const ModelView &m, const Charges &q, double &energy) {
  const std::size_t n = m.coupling.size1();
  if (q.size() != n)
    return false;
  std::vector<double> v(n);
  energy = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (q[i] < -1 || q[i] > 1)
      return false;
    const double b = m.external[i] + m.fixed[i];
    v[i] = -b;
    energy += b * q[i];
    for (std::size_t j = 0; j < n; ++j) {
      v[i] -= m.coupling(i, j) * q[j];
      energy += .5 * q[i] * m.coupling(i, j) * q[j];
    }
    const double x = v[i] + m.mu;
    if (!((q[i] == -1 && x < m.epsilon) ||
          (q[i] == 1 && x - m.eta > -m.epsilon) ||
          (q[i] == 0 && x > -m.epsilon && x - m.eta < m.epsilon)))
      return false;
  }
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      if (q[i] < q[j] && -v[i] + v[j] - m.coupling(i, j) < -m.epsilon)
        return false;
  return std::isfinite(energy);
}
Callbacks callbacks(const ModelView &m) {
  Callbacks cb;
  cb.validate = [&m](const Charges &q, double &e) { return full(m, q, e); };
  return cb;
}
Candidate candidate(const ModelView &m, const Charges &q) {
  double e;
  REQUIRE(full(m, q, e));
  return Candidate(q, e);
}
bool equal(const Charges &a, const Charges &b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}
double exhaustive(const ModelView &m, int radix) {
  std::size_t count = 1;
  for (std::size_t i = 0; i < m.coupling.size1(); ++i)
    count *= radix;
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t code = 0; code < count; ++code) {
    Charges q(m.coupling.size1());
    auto value = code;
    for (std::size_t i = 0; i < q.size(); ++i) {
      q[i] = static_cast<int>(value % radix) - 1;
      value /= radix;
    }
    double e;
    if (full(m, q, e))
      best = std::min(best, e);
  }
  return best;
}
} // namespace

TEST_CASE(
    "K6 refinement reaches the exhaustive ternary minimum with both fields") {
  Fixture f(3);
  f.domains.assign(6, 7);
  auto m = f.model();
  Options o;
  o.mode = Mode::K6;
  Geometry g(m, o, 73);
  auto initial = candidate(m, f.aligned(1));
  const std::vector<Candidate> original{initial, initial};
  auto cb = callbacks(m);
  std::atomic<unsigned> checks(0);
  cb.validate = [&](const Charges &q, double &e) {
    ++checks;
    return full(m, q, e);
  };
  auto result = run(m, g, original, o, 4, cb);
  REQUIRE(checks == original.size() + result.stats.trials);
  REQUIRE(result.improved);
  REQUIRE(g.clusterCount() == 1);
  REQUIRE(g.patternCount() == 729);
  REQUIRE(result.candidate.valid);
  REQUIRE(result.candidate.energy == Approx(exhaustive(m, 3)).margin(1e-12));
  REQUIRE(result.candidate.energy < initial.energy);
  REQUIRE(result.stats.distinct == 1);
  REQUIRE(result.stats.selected == 1);
  REQUIRE(result.stats.improvements > 0);
  REQUIRE(equal(original[0].config, initial.config));
}
TEST_CASE(
    "Runtime K10 and shared refinement preserve inputs and verified minima") {
  Fixture f(5);
  auto m = f.model();
  Options fixed;
  fixed.mode = Mode::K10;
  fixed.rounds = 4;
  Geometry gf(m, fixed, 91);
  auto initial = candidate(m, f.aligned(1));
  // Prove positive charges inadmissible before binary exhaustive enumeration.
  for (std::size_t i = 0; i < f.a.size1(); ++i) {
    double upper = -f.external[i] - f.fixed[i];
    for (std::size_t j = 0; j < f.a.size2(); ++j)
      upper += f.a(i, j);
    REQUIRE(upper + m.mu - m.eta < -m.epsilon);
  }
  std::vector<Candidate> inputs{initial, initial};
  auto one = run(m, gf, inputs, fixed, 1, callbacks(m));
  Options shared = fixed;
  shared.mode = Mode::SharedK10;
  Geometry gs(m, shared, 91);
  auto many = run(m, gs, inputs, shared, 64, callbacks(m));
  REQUIRE(gf.patternCount() == 1024);
  REQUIRE(many.candidate.valid);
  REQUIRE(many.candidate.energy == Approx(exhaustive(m, 2)).margin(1e-12));
  REQUIRE(many.candidate.energy <= one.candidate.energy);
  REQUIRE(gs.logicalBytes() <= 2 * shared.geometry_byte_cap);
  REQUIRE(equal(inputs[0].config, initial.config));
}
TEST_CASE("Refinement caps preserve lowest verified incumbent even with stale "
          "metadata") {
  Fixture f(3);
  auto m = f.model();
  auto high = candidate(m, f.aligned(1));
  auto low = candidate(m, f.aligned(0));
  high.energy = -100;
  low.energy = 100;
  std::vector<Candidate> inputs{high, low};
  Options o;
  o.mode = Mode::K6;
  SECTION("Geometry header cannot fit") { o.geometry_byte_cap = 1; }
  SECTION("Dedup header cannot fit") { o.dedup_byte_cap = 1; }
  Geometry g(m, o, 7);
  auto result = run(m, g, inputs, o, 4, callbacks(m));
  REQUIRE(result.stats.budget_exhausted);
  REQUIRE_FALSE(result.improved);
  REQUIRE(result.candidate.valid);
  REQUIRE(equal(result.candidate.config, low.config));
  double expected;
  REQUIRE(full(m, low.config, expected));
  REQUIRE(result.candidate.energy == expected);
  REQUIRE(g.logicalBytes() <= o.geometry_byte_cap);
}
TEST_CASE(
    "K10 ternary pattern cap is a budget skip and never a false success") {
  Fixture f(5);
  f.domains.assign(10, 7);
  auto m = f.model();
  Options o;
  o.mode = Mode::K10;
  Geometry g(m, o, 3);
  auto initial = candidate(m, f.aligned(1));
  REQUIRE(g.skipped());
  REQUIRE(g.budgetExhausted());
  REQUIRE(g.patternCount() == 0);
  auto result = run(m, g, {initial}, o, 1, callbacks(m));
  REQUIRE(result.stats.geometry_skipped);
  REQUIRE(result.stats.budget_exhausted);
  REQUIRE(equal(result.candidate.config, initial.config));
}
TEST_CASE("Stable energy ties retain ordinary candidate and ignore duplicate "
          "inputs") {
  Fixture f(3);
  f.external.clear();
  f.fixed.clear();
  auto m = f.model();
  Options o;
  o.mode = Mode::K6;
  Geometry g(m, o, 1);
  auto first = candidate(m, f.aligned(1)), second = candidate(m, f.aligned(0));
  REQUIRE(first.energy == second.energy);
  auto result = run(m, g, {first, second, first}, o, 2, callbacks(m));
  REQUIRE(result.stats.distinct == 2);
  REQUIRE(result.stats.selected == 2);
  REQUIRE(result.stats.improvements == 0);
  REQUIRE(equal(result.candidate.config, first.config));
}
TEST_CASE(
    "Full validation distrusts repair flags and recomputes returned energy") {
  Fixture f(3);
  auto m = f.model();
  Options o;
  o.mode = Mode::K6;
  Geometry g(m, o, 1);
  auto initial = candidate(m, f.aligned(1));
  auto cb = callbacks(m);
  SECTION("Forged valid flag cannot admit an invalid population") {
    cb.repair = [](const Charges &q) {
      Charges bad(q.size());
      bad.clear();
      return Candidate(bad, -1e20, true);
    };
    auto result = run(m, g, {initial}, o, 2, cb);
    REQUIRE(result.stats.trials > 0);
    REQUIRE(result.stats.improvements == 0);
    REQUIRE(equal(result.candidate.config, initial.config));
  }
  SECTION("Forged energy is replaced by the full Hamiltonian") {
    cb.repair = [](const Charges &q) { return Candidate(q, -1e20, true); };
    auto result = run(m, g, {initial}, o, 2, cb);
    double e;
    REQUIRE(full(m, result.candidate.config, e));
    REQUIRE(result.candidate.energy == e);
    REQUIRE(result.candidate.energy < initial.energy);
  }
  SECTION("Repair exhaustion is observable while retaining the incumbent") {
    cb.repair = [](const Charges &) {
      Candidate x;
      x.budget_exhausted = true;
      return x;
    };
    auto result = run(m, g, {initial}, o, 2, cb);
    REQUIRE(result.stats.budget_exhausted);
    REQUIRE(equal(result.candidate.config, initial.config));
  }
}
TEST_CASE(
    "Refinement rejects a geometry from another model and invalid options") {
  Fixture f(3), other(3);
  auto m = f.model(), wrong = other.model();
  Options o;
  o.mode = Mode::K6;
  Geometry g(m, o, 1);
  auto initial = candidate(m, f.aligned(1));
  REQUIRE_THROWS_AS(run(wrong, g, {initial}, o, 1, callbacks(wrong)),
                    std::invalid_argument);
  auto invalid = o;
  invalid.candidates = 33;
  REQUIRE_THROWS_AS(Geometry(m, invalid, 1), std::invalid_argument);
  invalid = o;
  invalid.rounds = 9;
  REQUIRE_THROWS_AS(Geometry(m, invalid, 1), std::invalid_argument);
  REQUIRE_THROWS_AS(run(m, g, {initial}, o, 0, callbacks(m)),
                    std::invalid_argument);
  Callbacks absent;
  REQUIRE_THROWS_AS(run(m, g, {initial}, o, 1, absent), std::invalid_argument);
}
TEST_CASE("Refinement handles malformed or nonfinite models without cached "
          "proposals") {
  Fixture f(3);
  Options o;
  o.mode = Mode::K6;
  SECTION("Nonfinite coupling") {
    f.a(0, 1) = std::numeric_limits<double>::infinity();
  }
  SECTION("Asymmetric coupling") { f.a(0, 1) += .1; }
  SECTION("Empty final domain") { f.domains[0] = 0; }
  SECTION("Nonfinite applied field") {
    f.external[0] = std::numeric_limits<double>::quiet_NaN();
  }
  auto m = f.model();
  Geometry g(m, o, 1);
  REQUIRE(g.skipped());
  REQUIRE(g.patternCount() == 0);
}
TEST_CASE("Worker exceptions propagate after joins without mutating input") {
  Fixture f(3);
  auto m = f.model();
  Options o;
  o.mode = Mode::K6;
  Geometry g(m, o, 1);
  auto initial = candidate(m, f.aligned(1));
  std::vector<Candidate> inputs{initial};
  auto cb = callbacks(m);
  cb.repair = [](const Charges &) -> Candidate {
    throw std::runtime_error("repair failure");
  };
  REQUIRE_THROWS_AS(run(m, g, inputs, o, 4, cb), std::runtime_error);
  REQUIRE(equal(inputs[0].config, initial.config));
  auto retried = run(m, g, inputs, o, 4, callbacks(m));
  REQUIRE(retried.candidate.valid);
  REQUIRE(retried.candidate.energy < initial.energy);
}

TEST_CASE(
    "Refinement normalizes stale false flags and supports candidate limits") {
  Fixture f(3);
  auto m = f.model();
  auto initial = candidate(m, f.aligned(1));
  initial.valid = false;
  initial.energy = -100;
  for (int count : {1, 8, 32}) {
    Options o;
    o.mode = Mode::K6;
    o.candidates = count;
    Geometry g(m, o, 73);
    auto result = run(m, g, {initial}, o, 64, callbacks(m));
    REQUIRE(result.candidate.valid);
    REQUIRE(result.improved);
    REQUIRE(result.stats.selected == 1);
  }
}
