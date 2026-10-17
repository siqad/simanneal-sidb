// Research-only exact validity-filtered block descent. No target/certificate input.
#ifndef SIMANNEAL_VBLOCK_H
#define SIMANNEAL_VBLOCK_H
#include "mv_kernel.h"
#include <array>
#include <chrono>
#include <cstring>
#include <map>
#include <limits>
#include <set>
#include <string>

namespace phys { namespace vblock {
using Charges = std::vector<int>;
using Fields = std::vector<double>;
constexpr double EPS = 1e-9, TOL = 2e-6;

inline bool population(int q, double f, double mu) {
  const double v = -f + mu;
  return (q == -1 && v < TOL) || (q == 1 && v - .59 > -TOL) ||
         (q == 0 && v > -TOL && v - .59 < TOL);
}

struct Budget {
  using Clock = std::chrono::steady_clock;
  Clock::time_point start = Clock::now();
  double max_ms;
  std::uint64_t max_blocks, blocks = 0;
  bool hit = false;
  explicit Budget(const mv::Options &o) : max_ms(o.vblock_max_ms), max_blocks(o.vblock_max_blocks) {}
  double ms() const { return std::chrono::duration<double, std::milli>(Clock::now()-start).count(); }
  bool expired() {
    if (max_ms > 0 && ms() >= max_ms) hit = true;
    return hit;
  }
  bool beginBlock() {
    if (max_blocks && blocks >= max_blocks) hit = true;
    if (expired()) return false;
    ++blocks;
    return true;
  }
};

struct Row {
  std::array<int, 10> z{};
  std::array<double, 10> bf{};
  double internal = 0;
  int index = 0;
};
struct Block {
  Charges sites;
  std::vector<Charges> allowed;
  std::vector<Row> table;
  bool cached = false;
};
struct Descent {
  Charges q; std::uint64_t moves;
  Descent(const Charges &state, std::uint64_t count = 0) : q(state), moves(count) {}
};

// Full keys avoid hash collisions. Field bits preserve incremental FP history,
// including signed zero; equal charges alone do not imply an equal next move.
using StateKey = std::pair<Charges, std::vector<std::uint64_t>>;
using Memo = std::map<StateKey, Descent>;
inline StateKey stateKey(const Charges &q, const Fields &f) {
  StateKey key{q, std::vector<std::uint64_t>(f.size())};
  static_assert(sizeof(double) == sizeof(std::uint64_t), "FP64 cache required");
  std::memcpy(key.second.data(), f.data(), f.size()*sizeof(double));
  return key;
}

class Geometry {
  const mv::ublas::matrix<double> &A;
  Fields H;
  double mu;
  int N;
  std::vector<Block> bank;

  // Match ptbench12.masks_of, including its fallback on an empty domain.
  std::vector<int> masks() const {
    std::vector<int> masks(N, 7);
    for (;;) {
      std::vector<int> next(masks);
      for (int i = 0; i < N; ++i) {
        double lo = H[i]+mu, hi = lo;
        for (int j = 0; j < N; ++j) {
          const int mn = masks[j]&1 ? -1 : masks[j]&2 ? 0 : 1;
          const int mx = masks[j]&4 ? 1 : masks[j]&2 ? 0 : -1;
          lo -= A(i,j)*mx; hi -= A(i,j)*mn;
        }
        int m = 0;
        if ((masks[i]&1) && lo <= 1e-9) m |= 1;
        if ((masks[i]&2) && lo <= .59+1e-9 && hi >= -1e-9) m |= 2;
        if ((masks[i]&4) && hi >= .59-1e-9) m |= 4;
        if (!m) return std::vector<int>(N,7);
        next[i] = m;
      }
      if (next == masks) return masks;
      masks.swap(next);
    }
  }

  // Lazy ternary tables are shared by all endpoints and repeated invocations.
  bool table(Block &b, Budget &budget) {
    if (b.cached) return true;
    const int k = b.sites.size();
    std::vector<Row> rows;
    Row assignment;
    std::uint64_t leaves = 0;
    // Enumerate only allowed assignments, retaining ascending ternary codes.
    std::function<void(int,int)> fill = [&](int depth, int code) {
      if (budget.hit) return;
      if (depth >= 0) {
        int power = 1; for (int i = 0; i < depth; ++i) power *= 3;
        for (int z : b.allowed[depth]) {
          assignment.z[depth] = z;
          fill(depth-1,code+(z+1)*power);
        }
        return;
      }
      if ((++leaves & 255) == 0 && budget.expired()) return;
      Row r; r.z = assignment.z; r.index = code;
      for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) r.bf[i] += r.z[j]*A(b.sites[j],b.sites[i]);
        r.internal += .5*r.z[i]*r.bf[i];
      }
      rows.push_back(r);
    };
    fill(k-1,0);
    if (budget.expired()) return false;
    b.table.swap(rows); b.cached = true;
    return true;
  }

public:
#ifdef SIMANNEAL_VBLOCK_DIAGNOSTIC
  const std::vector<Block> &blocks() const { return bank; }
  std::function<void(const Charges &, int, int, double, bool)> audit;
  std::uint64_t cache_hits = 0, states_scored = 0, checks = 0;
  double table_ms = 0, validation_ms = 0;
#endif
  Geometry(const mv::ublas::matrix<double> &a, const mv::ublas::vector<double> &ext,
           const mv::ublas::vector<double> &fixed, double chemical,
           const mv::Options &o, const std::vector<Charges> &knn)
      : A(a), H(a.size1()), mu(chemical), N(a.size1()) {
    for (int i = 0; i < N; ++i) H[i] = ext[i]+fixed[i];
    std::vector<std::vector<bool>> graph(N,std::vector<bool>(N,false));
    for (int i = 0; i < N; ++i) {
      Charges nb;
      if (o.vblock_graph == 1) nb = knn[i];
      else {
        for (int j = 0; j < N; ++j) if (j != i) nb.push_back(j);
        std::stable_sort(nb.begin(), nb.end(), [&](int x, int y) { return std::abs(A(i,x)) > std::abs(A(i,y)); });
        if (nb.size() > 8) nb.resize(8);
      }
      for (int j : nb) if (i != j) graph[i][j] = graph[j][i] = true;
    }
    auto grow = [&](int seed, int size, const Charges &excluded) {
      Charges s{seed};
      while (static_cast<int>(s.size()) < size) {
        int best = -1; double weight = -1;
        for (int j = 0; j < N; ++j) {
          if (std::find(s.begin(),s.end(),j) != s.end() ||
              std::find(excluded.begin(),excluded.end(),j) != excluded.end()) continue;
          bool connected = false; double w = 0;
          for (int i : s) { connected = connected || graph[j][i]; w += std::abs(A(j,i)); }
          if (connected && w > weight) { best = j; weight = w; }
        }
        if (best < 0) break;
        s.push_back(best);
      }
      return s;
    };
    const auto M = masks();
    std::set<Charges> seen;
    auto append = [&](const Charges &sites) {
      Charges key = sites; std::sort(key.begin(),key.end());
      if (!seen.insert(key).second) return;
      Block b; b.sites = sites;
      for (int i : sites) {
        Charges allowed;
        for (int z = -1; z <= 1; ++z) if (M[i] & (1 << (z+1))) allowed.push_back(z);
        b.allowed.push_back(allowed);
      }
      bank.push_back(std::move(b));
    };
    for (int size = 3; size <= std::min(o.vblock_k,N); ++size)
      for (int seed = 0; seed < N; ++seed) {
        const auto s = grow(seed,size,{});
        if (static_cast<int>(s.size()) == size) append(s);
      }
    const int lobe = o.vblock_k >= 8 ? 4 : o.vblock_k/2;
    if (N >= 2*lobe) for (int seed = 0; seed < N; ++seed) {
      auto s = grow(seed,lobe,{});
      if (static_cast<int>(s.size()) != lobe) continue;
      int other = -1; double weight = std::numeric_limits<double>::infinity();
      for (int j = 0; j < N; ++j) {
        if (std::find(s.begin(),s.end(),j) != s.end()) continue;
        double w = 0; for (int i : s) w += std::abs(A(j,i));
        if (w < weight) { other = j; weight = w; }
      }
      if (other < 0) continue;
      const auto t = grow(other,lobe,s);
      if (static_cast<int>(t.size()) == lobe) { s.insert(s.end(),t.begin(),t.end()); append(s); }
    }
  }

  Fields fields(const Charges &q) const {
    Fields f = H;
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) f[i] += A(i,j)*q[j];
    return f;
  }
  double energy(const Charges &q) const {
    double e = 0;
    for (int i = 0; i < N; ++i) {
      e += q[i]*H[i];
      for (int j = 0; j < N; ++j) e += .5*q[i]*A(i,j)*q[j];
    }
    return e;
  }
  bool valid(const Charges &q, const Fields &f) const {
    for (int i = 0; i < N; ++i) if (!population(q[i],f[i],mu)) return false;
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j)
      if (q[i] < q[j] && f[i]-f[j]-A(i,j) < -TOL) return false;
    return true;
  }

  Descent descend(const Charges &pre, Budget &budget, Memo *memo = nullptr) {
    Descent out{pre,0};
    Fields f = fields(pre);
    if (!valid(pre,f)) return out;
    std::vector<std::pair<StateKey,std::uint64_t>> visited;
    for (;;) {
      if (memo) {
        auto key = stateKey(out.q,f);
        const auto cached = memo->find(key);
        if (cached != memo->end()) {
#ifdef SIMANNEAL_VBLOCK_DIAGNOSTIC
          ++cache_hits;
#endif
          out.q = cached->second.q; out.moves += cached->second.moves;
          break;
        }
        visited.emplace_back(std::move(key),out.moves);
      }
      double bound = -EPS;
      Charges winner; Fields winner_f;
      for (auto &b : bank) {
        if (!budget.beginBlock()) break;
        const int k = b.sites.size();
        std::array<double,10> h{};
        double old = 0;
        for (int i = 0; i < k; ++i) {
          h[i] = f[b.sites[i]];
          double internal_f = 0;
          for (int j = 0; j < k; ++j) internal_f += A(b.sites[i],b.sites[j])*out.q[b.sites[j]];
          h[i] -= internal_f;
          old += out.q[b.sites[i]]*(h[i]+.5*internal_f);
        }
        struct Candidate { Row row; double de; };
        std::vector<Candidate> candidates;
        auto consider = [&](const Row &r, double e, bool population_checked = false) {
          if (!(e-old < bound)) return;
          bool differs = false;
          for (int i = 0; i < k; ++i) {
            differs = differs || r.z[i] != out.q[b.sites[i]];
            // Necessary block population filter avoids most full N^2 checks.
#ifndef SIMANNEAL_VBLOCK_NO_POP_PREFILTER
            if (!population_checked && !population(r.z[i],h[i]+r.bf[i],mu)) return;
#endif
          }
          if (differs) candidates.push_back(Candidate{r,e-old});
        };
        if (k <= 8) {
#ifdef SIMANNEAL_VBLOCK_DIAGNOSTIC
          const auto table_start = Budget::Clock::now();
#endif
          if (!table(b,budget)) break;
#ifdef SIMANNEAL_VBLOCK_DIAGNOSTIC
          table_ms += std::chrono::duration<double,std::milli>(Budget::Clock::now()-table_start).count();
          states_scored += b.table.size();
#endif
          int count = 0;
          for (const auto &r : b.table) {
            if ((++count & 255) == 0 && budget.expired()) break;
#ifndef SIMANNEAL_VBLOCK_NO_POP_PREFILTER
            // Cached block fields reject rows before their energy dot product.
            bool possible = true;
            for (int i = 0; i < k; ++i)
              if (!population(r.z[i],h[i]+r.bf[i],mu)) { possible = false; break; }
            if (!possible) continue;
#endif
            double e = r.internal;
            for (int i = 0; i < k; ++i) e += r.z[i]*h[i];
            consider(r,e,true);
          }
        } else {
          // Exact depth-first B&B: independent linear/pair lower bounds and
          // conservative assigned-site population intervals, as in the reference.
          Row r; std::uint64_t nodes = 0;
          std::function<void(int,double,int)> visit = [&](int depth, double partial, int index) {
            if ((++nodes & 255) == 0 && budget.expired()) return;
            if (budget.hit) return;
            if (depth == k) {
              r.index = index;
              for (int i = 0; i < k; ++i) {
                r.bf[i] = 0;
                for (int j = 0; j < k; ++j) r.bf[i] += r.z[j]*A(b.sites[j],b.sites[i]);
              }
              consider(r,partial); return;
            }
            for (int z : b.allowed[depth]) {
              r.z[depth] = z;
              double linear = h[depth];
              for (int i = 0; i < depth; ++i) linear += r.z[i]*A(b.sites[i],b.sites[depth]);
              const double p = partial+z*linear;
              double lower = p;
              for (int j = depth+1; j < k; ++j) {
                double a = h[j];
                for (int i = 0; i <= depth; ++i) a += r.z[i]*A(b.sites[i],b.sites[j]);
                double term = std::numeric_limits<double>::infinity();
                for (int t : b.allowed[j]) term = std::min(term,a*t);
                lower += term;
                for (int l = j+1; l < k; ++l) {
                  term = std::numeric_limits<double>::infinity();
                  for (int t : b.allowed[j]) for (int u : b.allowed[l]) term = std::min(term,t*u*A(b.sites[j],b.sites[l]));
                  lower += term;
                }
              }
              if (!(lower-old < bound+1e-10*(1+std::abs(old)+std::abs(lower)))) continue;
              bool possible = true;
              for (int i = 0; i <= depth && possible; ++i) {
                double lo = h[i];
                for (int j = 0; j <= depth; ++j) lo += r.z[j]*A(b.sites[j],b.sites[i]);
                double hi = lo;
                for (int j = depth+1; j < k; ++j) {
                  double mn = std::numeric_limits<double>::infinity(), mx = -mn;
                  for (int t : b.allowed[j]) { const double x = t*A(b.sites[i],b.sites[j]); mn = std::min(mn,x); mx = std::max(mx,x); }
                  lo += mn; hi += mx;
                }
                possible = (r.z[i] == -1 && hi > mu-TOL) || (r.z[i] == 1 && lo < mu-.59+TOL) ||
                           (r.z[i] == 0 && lo < mu+TOL && hi > mu-.59-TOL);
              }
              if (possible) visit(depth+1,p,index*3+z+1);
            }
          };
          visit(0,0,0);
        }
        if (budget.expired()) break; // Discard an incomplete block, retain earlier winners.
        std::sort(candidates.begin(),candidates.end(), [](const Candidate &a, const Candidate &b) {
          return a.de < b.de || (a.de == b.de && a.row.index < b.row.index);
        });
        for (const auto &c : candidates) {
          if (budget.expired()) break;
          Charges q = out.q; Fields fn = f;
          for (int i = 0; i < k; ++i) {
            const int d = c.row.z[i]-q[b.sites[i]];
            q[b.sites[i]] = c.row.z[i];
            if (d) for (int j = 0; j < N; ++j) fn[j] += A(j,b.sites[i])*d;
          }
#ifdef SIMANNEAL_VBLOCK_DIAGNOSTIC
          const auto valid_start = Budget::Clock::now(); ++checks;
#endif
          const bool accepted = valid(q,fn);
#ifdef SIMANNEAL_VBLOCK_DIAGNOSTIC
          validation_ms += std::chrono::duration<double,std::milli>(Budget::Clock::now()-valid_start).count();
          if (audit) audit(out.q, static_cast<int>(&b-bank.data()), c.row.index, c.de, accepted);
#endif
          if (accepted) { bound = c.de; winner.swap(q); winner_f.swap(fn); break; }
        }
        if (budget.hit) break;
      }
      if (winner.empty()) break;
      out.q.swap(winner); f.swap(winner_f); ++out.moves;
      if (budget.hit) break;
    }
    if (memo && !budget.hit)
      for (auto &entry : visited)
        memo->emplace(std::move(entry.first),Descent(out.q,out.moves-entry.second));
    return out;
  }
};
} }
#endif
