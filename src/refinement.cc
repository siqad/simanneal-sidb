#include "refinement.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>

namespace phys {
namespace refinement {
namespace {
const std::size_t hard_bytes = 8 * 1024 * 1024, hard_entries = 4096;
void checkOptions(const Options &o) {
  if (o.mode != Mode::Disabled && o.mode != Mode::K6 && o.mode != Mode::K10 &&
      o.mode != Mode::SharedK10)
    throw std::invalid_argument("Unknown refinement mode");
  if (o.candidates < 1 || o.candidates > 32 || o.rounds < 1 || o.rounds > 8 ||
      o.trials < 1 || o.trials > 2)
    throw std::invalid_argument(
        "Refinement requires candidates 1..32, rounds 1..8 and trials 1..2");
  if (o.geometry_byte_cap > hard_bytes || o.dedup_byte_cap > hard_bytes)
    throw std::invalid_argument("Refinement logical cap exceeds 8 MiB");
}
bool validModel(const ModelView &m) {
  const std::size_t n = m.coupling.size1();
  if (!n || n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      m.coupling.size2() != n || m.external.size() != n ||
      m.fixed.size() != n || m.domains.size() != n || !std::isfinite(m.mu) ||
      !std::isfinite(m.eta) || m.eta < 0 || !std::isfinite(m.epsilon) ||
      m.epsilon < 0)
    return false;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(m.external[i]) || !std::isfinite(m.fixed[i]) ||
        !m.domains[i] || (m.domains[i] & ~7))
      return false;
    for (std::size_t j = 0; j < n; ++j) {
      const FPType a = m.coupling(i, j);
      if (!std::isfinite(a) || a < 0 || (i == j && a != 0) ||
          a != m.coupling(j, i))
        return false;
    }
  }
  return true;
}
bool chargeShape(const Charges &q, std::size_t n) {
  if (q.size() != n)
    return false;
  for (int x : q)
    if (x < -1 || x > 1)
      return false;
  return true;
}
Candidate validated(const Candidate &c, const ModelView &m,
                    const Callbacks &cb) {
  Candidate out;
  FPType energy;
  if (chargeShape(c.config, m.coupling.size1()) &&
      cb.validate(c.config, energy) && std::isfinite(energy))
    out = Candidate(c.config, energy);
  return out;
}
void addStats(Stats &a, const Stats &b) {
  a.patterns += b.patterns;
  a.lower += b.lower;
  a.trials += b.trials;
  a.improvements += b.improvements;
  a.budget_exhausted |= b.budget_exhausted;
}
template <int K> struct Pattern {
  std::array<std::int8_t, K> q;
  std::array<FPType, K> field;
  FPType internal = 0;
};
template <int K> struct Cluster {
  std::array<int, K> sites;
  std::vector<Pattern<K>> patterns;
};
template <int K> struct Cache {
  std::vector<Cluster<K>> clusters;
  std::size_t bytes = 0, patterns = 0;
  bool budget = false;
  Cache(const ModelView &m, std::size_t cap, int offset) {
    const int n = static_cast<int>(m.coupling.size1());
    if (n < K)
      return;
    const int count = std::min(32, n);
    // Account for persistent vector headers/capacity before allocating them.
    const std::size_t headers = sizeof(Cache<K>) + count * sizeof(Cluster<K>);
    if (headers > cap) {
      budget = true;
      return;
    }
    bytes = headers;
    clusters.reserve(count);
    for (int c = 0; c < count; ++c) {
      const int center = static_cast<int>(
          (static_cast<std::size_t>(c) * n / count + offset) % n);
      std::vector<int> order;
      order.reserve(n - 1);
      for (int j = 0; j < n; ++j)
        if (j != center)
          order.push_back(j);
      std::sort(order.begin(), order.end(), [&](int a, int b) {
        return m.coupling(center, a) > m.coupling(center, b) ||
               (m.coupling(center, a) == m.coupling(center, b) && a < b);
      });
      Cluster<K> cl;
      cl.sites[0] = center;
      for (int k = 1; k < K; ++k)
        cl.sites[k] = order[k - 1];
      std::sort(cl.sites.begin(), cl.sites.end());
      bool duplicate = false;
      for (const auto &old : clusters)
        duplicate |= old.sites == cl.sites;
      if (duplicate)
        continue;
      std::array<std::array<int, 3>, K> values;
      std::array<int, K> radix;
      int pattern_count = 1;
      for (int k = 0; k < K; ++k) {
        radix[k] = 0;
        for (int q = -1; q <= 1; ++q)
          if (m.domains[cl.sites[k]] & (1 << (q + 1)))
            values[k][radix[k]++] = q;
        pattern_count *= radix[k];
      }
      const std::size_t payload =
          static_cast<std::size_t>(pattern_count) * sizeof(Pattern<K>);
      if (pattern_count > 4096 || payload > cap - bytes) {
        budget = true;
        continue;
      }
      cl.patterns.reserve(pattern_count);
      for (int code = 0; code < pattern_count; ++code) {
        Pattern<K> p;
        int value = code;
        for (int k = K - 1; k >= 0; --k) {
          p.q[k] = values[k][value % radix[k]];
          value /= radix[k];
        }
        for (int i = 0; i < K; ++i) {
          p.field[i] = 0;
          for (int j = 0; j < K; ++j)
            p.field[i] += m.coupling(cl.sites[i], cl.sites[j]) * p.q[j];
        }
        for (int i = 0; i < K; ++i)
          for (int j = 0; j < i; ++j)
            p.internal +=
                m.coupling(cl.sites[i], cl.sites[j]) * p.q[i] * p.q[j];
        cl.patterns.push_back(p);
      }
      bytes += payload;
      patterns += cl.patterns.size();
      clusters.push_back(std::move(cl));
    }
  }
};
template <int K> struct Move {
  std::array<int, K> sites, q;
  FPType delta = 0;
};
template <int K> struct Selection {
  std::array<Move<K>, 2> moves;
  int count = 0;
  std::uint64_t patterns = 0, lower = 0;
};
template <int K> int movedCharge(const Move<K> &x, const Charges &q, int site) {
  for (int k = 0; k < K; ++k)
    if (x.sites[k] == site)
      return x.q[k];
  return q[site];
}
template <int K>
bool sameMove(const Move<K> &a, const Move<K> &b, const Charges &q) {
  for (int i : a.sites)
    if (movedCharge(a, q, i) != movedCharge(b, q, i))
      return false;
  for (int i : b.sites)
    if (movedCharge(a, q, i) != movedCharge(b, q, i))
      return false;
  return true;
}
template <int K>
Selection<K> select(const Cache<K> &g, const ModelView &m, const Charges &q,
                    const Potentials &v, int maximum) {
  Selection<K> out;
  for (const auto &cl : g.clusters) {
    std::array<FPType, K> h;
    FPType current = 0;
    for (int i = 0; i < K; ++i) {
      h[i] = -v[cl.sites[i]];
      for (int j = 0; j < K; ++j)
        h[i] -= m.coupling(cl.sites[i], cl.sites[j]) * q[cl.sites[j]];
      current += h[i] * q[cl.sites[i]];
    }
    for (int i = 0; i < K; ++i)
      for (int j = 0; j < i; ++j)
        current += m.coupling(cl.sites[i], cl.sites[j]) * q[cl.sites[i]] *
                   q[cl.sites[j]];
    for (const auto &p : cl.patterns) {
      ++out.patterns;
      std::array<FPType, K> local;
      bool valid = true;
      for (int i = 0; i < K; ++i) {
        local[i] = -h[i] - p.field[i];
        const FPType x = local[i] + m.mu;
        if (!(std::isfinite(local[i]) &&
              ((p.q[i] == -1 && x < m.epsilon) ||
               (p.q[i] == 1 && x - m.eta > -m.epsilon) ||
               (p.q[i] == 0 && x > -m.epsilon && x - m.eta < m.epsilon)))) {
          valid = false;
          break;
        }
      }
      if (!valid)
        continue;
      FPType e = p.internal;
      for (int i = 0; i < K; ++i)
        e += h[i] * p.q[i];
      const FPType delta = e - current;
      if (!std::isfinite(delta) || !(delta < -1e-8))
        continue;
      for (int i = 0; i < K && valid; ++i)
        for (int j = 0; j < K; ++j)
          if (p.q[i] < p.q[j] &&
              -local[i] + local[j] - m.coupling(cl.sites[i], cl.sites[j]) <
                  -m.epsilon) {
            valid = false;
            break;
          }
      if (!valid)
        continue;
      ++out.lower;
      Move<K> move;
      move.sites = cl.sites;
      move.delta = delta;
      for (int i = 0; i < K; ++i)
        move.q[i] = p.q[i];
      bool duplicate = false;
      for (int i = 0; i < out.count; ++i)
        duplicate |= sameMove(move, out.moves[i], q);
      if (duplicate)
        continue;
      int pos = 0;
      while (pos < out.count && !(delta < out.moves[pos].delta))
        ++pos;
      if (pos >= maximum)
        continue;
      const int end = std::min(out.count, maximum - 1);
      for (int i = end; i > pos; --i)
        out.moves[i] = out.moves[i - 1];
      out.moves[pos] = move;
      out.count = std::min(maximum, out.count + 1);
    }
  }
  return out;
}
template <int K>
Result quench(const Cache<K> &g, const ModelView &m, const Candidate &initial,
              const Options &o, const Callbacks &cb) {
  Result out;
  out.candidate = initial;
  for (int round = 0; round < o.rounds; ++round) {
    const Candidate current = out.candidate;
    const auto before = out.stats.improvements;
    Potentials v(m.coupling.size1());
    bool finite = true;
    for (std::size_t i = 0; i < v.size(); ++i) {
      v[i] = -(m.external[i] + m.fixed[i]);
      for (std::size_t j = 0; j < v.size(); ++j)
        v[i] -= m.coupling(i, j) * current.config[j];
      finite &= std::isfinite(v[i]);
    }
    if (!finite)
      break;
    const auto selected = select(g, m, current.config, v, o.trials);
    out.stats.patterns += selected.patterns;
    out.stats.lower += selected.lower;
    Charges trial(current.config);
    for (int t = 0; t < selected.count; ++t) {
      trial = current.config;
      for (int k = 0; k < K; ++k)
        trial[selected.moves[t].sites[k]] = selected.moves[t].q[k];
      ++out.stats.trials;
      Candidate proposed =
          cb.repair ? cb.repair(trial) : Candidate(trial, 0, true);
      out.stats.budget_exhausted |= proposed.budget_exhausted;
      proposed = validated(proposed, m, cb);
      if (proposed.valid && proposed.energy < out.candidate.energy) {
        out.candidate = std::move(proposed);
        ++out.stats.improvements;
      }
    }
    if (out.stats.improvements == before)
      break;
  }
  return out;
}
struct Engine {
  virtual ~Engine() {}
  virtual Result refine(const ModelView &, const Candidate &, const Options &,
                        const Callbacks &) const = 0;
  std::size_t bytes = 0, clusters = 0, patterns = 0, geometry_count = 0;
  bool budget = false;
};
template <int K> struct TypedEngine : Engine {
  std::vector<Cache<K>> caches;
  TypedEngine(const ModelView &m, const Options &o, int offset) {
    caches.emplace_back(m, o.geometry_byte_cap, 0);
    if (offset != 0)
      caches.emplace_back(m, o.geometry_byte_cap, offset);
    for (const auto &g : caches) {
      bytes += g.bytes;
      clusters += g.clusters.size();
      patterns += g.patterns;
      geometry_count += !g.clusters.empty();
      budget |= g.budget;
    }
  }
  Result refine(const ModelView &m, const Candidate &initial, const Options &o,
                const Callbacks &cb) const override {
    Result best;
    best.candidate = initial;
    for (const auto &g : caches) {
      // Each geometry starts independently from the exact same charge vector.
      Result x = quench(g, m, initial, o, cb);
      addStats(best.stats, x.stats);
      if (x.candidate.valid && x.candidate.energy < best.candidate.energy)
        best.candidate = std::move(x.candidate);
    }
    return best;
  }
};
struct VerifiedInput {
  std::size_t index;
  FPType energy;
};
struct DedupEntry {
  std::vector<int> key;
  std::uint64_t hash;
};
std::uint64_t chargeHash(const Charges &q) {
  std::uint64_t h = UINT64_C(1469598103934665603);
  for (int x : q) {
    h ^= static_cast<unsigned>(x + 1);
    h *= UINT64_C(1099511628211);
  }
  return h;
}
// mt19937_64 has a specified stream; library integer distributions do not.
// Rejection of the low incomplete residue range makes modulo reduction
// unbiased.
int sharedOffset(std::uint64_t seed, int n) {
  const int step = n / std::min(32, n);
  if (step < 2)
    return 0;
  const std::uint64_t bound = static_cast<std::uint64_t>(step - 1);
  const std::uint64_t threshold = (UINT64_C(0) - bound) % bound;
  std::mt19937_64 rng(seed ^ UINT64_C(0xd1b54a32d192ed03));
  std::uint64_t value;
  do {
    value = rng();
  } while (value < threshold);
  // Every fixed-center gap is at least step. A shift strictly inside that gap
  // produces a disjoint center set, without wrapping onto the fixed centers.
  return 1 + static_cast<int>(value % bound);
}
bool sameKey(const std::vector<int> &key, const Charges &q) {
  if (key.size() != q.size())
    return false;
  return std::equal(key.begin(), key.end(), q.begin());
}
} // namespace

struct Geometry::Impl {
  const Matrix *a;
  const Potentials *external, *fixed;
  const std::vector<unsigned char> *domains;
  FPType mu, eta, epsilon;
  Mode mode;
  std::size_t geometry_byte_cap;
  int offset = 0;
  bool shared_fallback = false;
  std::unique_ptr<Engine> engine;
  Impl(const ModelView &m, const Options &o, std::uint64_t seed)
      : a(&m.coupling), external(&m.external), fixed(&m.fixed),
        domains(&m.domains), mu(m.mu), eta(m.eta), epsilon(m.epsilon),
        mode(o.mode), geometry_byte_cap(o.geometry_byte_cap) {
    checkOptions(o);
    if (o.mode == Mode::Disabled || !validModel(m))
      return;
    if (o.mode == Mode::SharedK10) {
      offset = sharedOffset(seed, static_cast<int>(a->size1()));
      shared_fallback = offset == 0;
    }
    if (o.mode == Mode::K6)
      engine.reset(new TypedEngine<6>(m, o, 0));
    else
      engine.reset(new TypedEngine<10>(m, o, offset));
  }
  bool matches(const ModelView &m, const Options &o) const {
    return a == &m.coupling && external == &m.external && fixed == &m.fixed &&
           domains == &m.domains && mu == m.mu && eta == m.eta &&
           epsilon == m.epsilon && mode == o.mode &&
           geometry_byte_cap == o.geometry_byte_cap;
  }
};
Geometry::Geometry(const ModelView &m, const Options &o, std::uint64_t seed)
    : impl_(new Impl(m, o, seed)) {}
Geometry::~Geometry() = default;
std::size_t Geometry::logicalBytes() const {
  return impl_->engine ? impl_->engine->bytes : 0;
}
std::size_t Geometry::clusterCount() const {
  return impl_->engine ? impl_->engine->clusters : 0;
}
std::size_t Geometry::patternCount() const {
  return impl_->engine ? impl_->engine->patterns : 0;
}
int Geometry::centerOffset() const { return impl_->offset; }
std::size_t Geometry::geometryCount() const {
  return impl_->engine ? impl_->engine->geometry_count : 0;
}
bool Geometry::sharedSingleCacheFallback() const {
  return impl_->shared_fallback;
}
bool Geometry::budgetExhausted() const {
  return impl_->engine && impl_->engine->budget;
}
bool Geometry::skipped() const {
  return impl_->mode != Mode::Disabled && clusterCount() == 0;
}

Result run(const ModelView &m, const Geometry &g,
           const std::vector<Candidate> &original, const Options &o,
           int workers, const Callbacks &cb) {
  checkOptions(o);
  if (!g.impl_->matches(m, o))
    throw std::invalid_argument(
        "Refinement geometry belongs to another model/mode/budget");
  if (!cb.validate)
    throw std::invalid_argument("Refinement requires a full validator");
  if (workers < 1)
    throw std::invalid_argument("Refinement workers must be positive");
  Result result;
  result.stats.geometry_bytes = g.logicalBytes();
  result.stats.geometry_count = g.geometryCount();
  result.stats.shared_single_cache_fallback = g.sharedSingleCacheFallback();
  result.stats.budget_exhausted = g.budgetExhausted();
  result.stats.geometry_skipped = g.skipped();
  if (original.empty())
    return result;
  const std::size_t capacity = original.size();
  const bool bounded_count = capacity <= hard_entries;
  const std::size_t slot_count = bounded_count ? capacity * 2 : 0;
  const std::size_t headers = bounded_count
                                  ? sizeof(std::vector<VerifiedInput>) +
                                        capacity * sizeof(VerifiedInput) +
                                        sizeof(std::vector<DedupEntry>) +
                                        capacity * sizeof(DedupEntry) +
                                        sizeof(std::vector<std::size_t>) +
                                        slot_count * sizeof(std::size_t)
                                  : 0;
  const bool can_select = bounded_count && headers <= o.dedup_byte_cap &&
                          o.mode != Mode::Disabled && !g.skipped();
  std::vector<VerifiedInput> order;
  if (can_select)
    order.reserve(capacity);
  // Normalize each endpoint once. The private energy/index list is a proof for
  // this immutable input/model, not trust in public Candidate metadata.
  for (std::size_t i = 0; i < original.size(); ++i) {
    const auto &c = original[i];
    FPType energy;
    if (!chargeShape(c.config, m.coupling.size1()) ||
        !cb.validate(c.config, energy) || !std::isfinite(energy))
      continue;
    if (!result.candidate.valid || energy < result.candidate.energy)
      result.candidate = Candidate(c.config, energy);
    if (can_select)
      order.push_back(VerifiedInput{i, energy});
  }
  if (o.mode == Mode::Disabled || g.skipped() || !result.candidate.valid)
    return result;
  if (!can_select) {
    result.stats.budget_exhausted = true;
    return result;
  }
  const FPType fallback_energy = result.candidate.energy;
  std::stable_sort(order.begin(), order.end(),
                   [](const VerifiedInput &a, const VerifiedInput &b) {
                     return a.energy < b.energy;
                   });
  result.stats.dedup_bytes = headers;
  std::vector<DedupEntry> entries;
  entries.reserve(capacity);
  const std::size_t empty = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> slots(slot_count, empty);
  std::vector<Candidate> chosen;
  chosen.reserve(o.candidates);
  for (const auto &verified : order) {
    const auto &c = original[verified.index];
    const auto hash = chargeHash(c.config);
    std::size_t slot = hash % slot_count;
    bool duplicate = false;
    while (slots[slot] != empty) {
      const auto &e = entries[slots[slot]];
      if (e.hash == hash && sameKey(e.key, c.config)) {
        duplicate = true;
        break;
      }
      slot = (slot + 1) % slot_count;
    }
    if (duplicate)
      continue;
    const std::size_t payload = c.config.size() * sizeof(int);
    if (payload > o.dedup_byte_cap - result.stats.dedup_bytes) {
      result.stats.budget_exhausted = true;
      return result;
    }
    DedupEntry e;
    e.hash = hash;
    e.key.assign(c.config.begin(), c.config.end());
    slots[slot] = entries.size();
    entries.push_back(std::move(e));
    result.stats.dedup_bytes += payload;
    ++result.stats.distinct;
    if (chosen.size() < static_cast<std::size_t>(o.candidates)) {
      chosen.emplace_back(
          c.config, verified.energy); // Reuse private normalization proof.
    }
  }
  result.stats.selected = chosen.size();
  if (chosen.empty())
    return result;
  // Release dedup keys before allocating parallel refinement outputs.
  std::vector<DedupEntry>().swap(entries);
  std::vector<std::size_t>().swap(slots);
  std::vector<VerifiedInput>().swap(order);
  std::vector<Result> outputs(chosen.size());
  std::atomic<std::size_t> next(0);
  std::atomic<bool> failed(false);
  std::exception_ptr failure;
  std::mutex failure_mutex;
  auto worker = [&] {
    try {
      while (!failed.load()) {
        const auto i = next.fetch_add(1);
        if (i >= chosen.size())
          break;
        outputs[i] = g.impl_->engine->refine(m, chosen[i], o, cb);
      }
    } catch (...) {
      std::lock_guard<std::mutex> guard(failure_mutex);
      if (!failure)
        failure = std::current_exception();
      failed.store(true);
    }
  };
  std::vector<std::thread> threads;
  try {
    for (int i = 0; i < std::min<int>(workers, chosen.size()); ++i)
      threads.emplace_back(worker);
  } catch (...) {
    failed.store(true);
    for (auto &t : threads)
      if (t.joinable())
        t.join();
    throw;
  }
  for (auto &t : threads)
    t.join();
  if (failure)
    std::rethrow_exception(failure);
  // Stable input-index reduction retains the ordinary incumbent on energy ties.
  for (const auto &out : outputs) {
    addStats(result.stats, out.stats);
    if (out.candidate.valid && out.candidate.energy < result.candidate.energy)
      result.candidate = out.candidate;
  }
  result.improved =
      result.candidate.valid && result.candidate.energy < fallback_energy;
  return result;
}
} // namespace refinement
} // namespace phys
