// Bounded local refinement inspired by QuickExact and ClusterComplete.
// Heuristic fixed-outsider cluster refinement, not an exact solver.
// Copyright 2026 SiQAD contributors. Licensed under Apache-2.0.
#ifndef SIMANNEAL_REFINEMENT_H
#define SIMANNEAL_REFINEMENT_H
#include "global.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
namespace phys {
namespace refinement {
namespace ublas = boost::numeric::ublas;
using Charges = ublas::vector<int>;
using Matrix = ublas::matrix<FPType>;
using Potentials = ublas::vector<FPType>;
enum class Mode { Disabled, K6, K10, SharedK10 };
struct Options {
  Mode mode = Mode::Disabled;
  int candidates = 8; // Default Q8; supported range is 1..32.
  int rounds = 1;
  int trials = 1;
  std::size_t geometry_byte_cap = 8 * 1024 * 1024; // Per geometry.
  std::size_t dedup_byte_cap = 8 * 1024 * 1024;
};
// Referenced arrays belong to one immutable model and outlive Geometry/run.
struct ModelView {
  const Matrix &coupling;
  const Potentials &external;
  const Potentials &fixed;
  const std::vector<unsigned char> &domains; // Bits for -1, 0, +1.
  FPType mu, eta, epsilon;
  ModelView(const Matrix &a, const Potentials &b, const Potentials &f,
            const std::vector<unsigned char> &d, FPType m, FPType e,
            FPType tolerance)
      : coupling(a), external(b), fixed(f), domains(d), mu(m), eta(e),
        epsilon(tolerance) {}
};
struct Candidate {
  Charges config;
  FPType energy = 0;
  bool valid = false;
  bool budget_exhausted = false; // Optional bounded-repair outcome metadata.
  Candidate() = default;
  Candidate(const Charges &q, FPType e, bool v = true)
      : config(q), energy(e), valid(v) {}
};
struct Stats {
  std::uint64_t distinct = 0, selected = 0, patterns = 0, lower = 0;
  std::uint64_t trials = 0, improvements = 0;
  std::size_t geometry_bytes = 0, dedup_bytes = 0, geometry_count = 0;
  bool shared_single_cache_fallback = false;
  bool budget_exhausted = false, geometry_skipped = false;
};
struct Result {
  Candidate candidate;
  Stats stats;
  bool improved = false;
};
struct Callbacks {
  // Mandatory full validator and energy evaluator. Public valid flags are no
  // proof. Both callbacks must allow concurrent calls against the immutable
  // model.
  std::function<bool(const Charges &, FPType &)> validate;
  // Optional bounded deterministic repair; otherwise validate proposals
  // directly.
  std::function<Candidate(const Charges &)> repair;
};
// Immutable exact K6/K10 cache layouts. Caps include geometry/vector headers,
// reserved cluster headers and pattern payload; they are not total RSS limits.
class Geometry {
public:
  Geometry(const ModelView &, const Options &, std::uint64_t job_seed);
  ~Geometry();
  Geometry(const Geometry &) = delete;
  Geometry &operator=(const Geometry &) = delete;
  std::size_t logicalBytes() const;
  std::size_t clusterCount() const;
  std::size_t patternCount() const;
  int centerOffset() const;
  // Number of nonempty cached geometries, after pattern/cap skips.
  std::size_t geometryCount() const;
  // Intentional SharedK10 fallback when floor(N/min(32,N)) < 2.
  bool sharedSingleCacheFallback() const;
  bool budgetExhausted() const;
  bool skipped() const;

private:
  struct Impl;
  std::unique_ptr<const Impl> impl_;
  friend Result run(const ModelView &, const Geometry &,
                    const std::vector<Candidate> &, const Options &, int,
                    const Callbacks &);
};
// After ordinary workers join. Inputs are immutable; public energy/valid
// metadata is ignored. Inputs normalize once; private proofs cover selected
// starts. Every prospective replacement is independently validated.
// Fixed/shifted geometries refine identical inputs without any global-model
// swaps.
Result run(const ModelView &, const Geometry &, const std::vector<Candidate> &,
           const Options &, int workers, const Callbacks &);
} // namespace refinement
} // namespace phys
#endif
