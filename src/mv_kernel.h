// End-of-run cleanup from research/vblock-cpu 07c6e99. Annealing is unchanged.
#ifndef SIMANNEAL_MV_KERNEL_H
#define SIMANNEAL_MV_KERNEL_H
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
namespace phys { namespace mv {
namespace ublas = boost::numeric::ublas;
typedef double FP;
struct Options {
  int polish_mode = -1; // -1 Auto, 0 off, 1 relax, 2 single/hop, 3 pair/swap
  int polish_top = -1;  // -1 Auto, 0 all restarts, K best distinct endpoints
  int polish_rounds = 8;
  int knn = 8;
  void validate() const {
    if (polish_mode < 0 || polish_mode > 3 || polish_top < 0 || polish_rounds < 0 || knn < 0)
      throw std::invalid_argument("mv_polish options out of range");
  }
};
struct Stats {
  std::uint64_t polish_restarts = 0, polish_relax_fixed = 0,
                polish_improved = 0, polish_kick_evals = 0;
  double polish_delta_e = 0;
  void add(const Stats &o) {
    polish_restarts += o.polish_restarts; polish_relax_fixed += o.polish_relax_fixed;
    polish_improved += o.polish_improved; polish_kick_evals += o.polish_kick_evals;
    polish_delta_e += o.polish_delta_e;
  }
};

// Shared geometry: k nearest neighbours by distance.
inline std::vector<std::vector<int>> nearest(const ublas::matrix<FP> &r, int k) {
  const int n = static_cast<int>(r.size1());
  std::vector<std::vector<int>> out(n);
  for (int i = 0; i < n; ++i) {
    std::vector<int> idx;
    for (int j = 0; j < n; ++j) if (j != i) idx.push_back(j);
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int a, int b) { return r(i, a) < r(i, b); });
    if (k > 0 && static_cast<int>(idx.size()) > k) idx.resize(k);
    out[i] = idx;
  }
  return out;
}

// Local state view with incremental fields. n in {-1,0,1}; v = -(v_ext+v_fc) - V n.
struct State {
  const ublas::matrix<FP> *V = nullptr;
  ublas::vector<int> *n = nullptr;
  ublas::vector<FP> *v = nullptr;
  FP *E = nullptr;
  FP mu = 0, eta = 0;

  int size() const { return static_cast<int>(n->size()); }
  // Energy change for charge deltas d at sites s (no duplicates).
  FP dE(const int *s, const int *d, int m) const {
    FP e = 0;
    for (int a = 0; a < m; ++a) {
      e += -d[a] * (*v)[s[a]] + 0.5 * d[a] * d[a] * (*V)(s[a], s[a]);
      for (int b = a + 1; b < m; ++b) e += d[a] * d[b] * (*V)(s[a], s[b]);
    }
    return e;
  }
  // Chemical part of dG = mu*dQ + eta*dP with Q = N- - N+ = -sum n.
  FP dChem(const int *s, const int *d, int m) const {
    FP g = 0;
    for (int a = 0; a < m; ++a) {
      const int before = (*n)[s[a]], after = before + d[a];
      g += mu * (-d[a]) + eta * ((after == 1) - (before == 1));
    }
    return g;
  }
  void apply(const int *s, const int *d, int m, FP de) {
    const int count = size();
    for (int a = 0; a < m; ++a) {
      if (!d[a]) continue;
      (*n)[s[a]] += d[a];
      const FP *row = &(V->data()[static_cast<std::size_t>(s[a]) * count]);
      FP *loc = &(*v)[0];
      for (int k = 0; k < count; ++k) loc[k] -= row[k] * d[a];
    }
    *E += de;
  }
};

// Steepest descent to population + configuration stability: single-site
// changes ranked by dG, electron hops (validator hop set: donor charge <
// acceptor charge) ranked by dE. Every accepted step strictly lowers G, so
// the descent terminates. If region is non-null, only moves whose sites all
// lie in region are considered (local relax; caller must validate globally).
inline bool relax(State &st, int cap, FP tol = 1e-9, const std::vector<int> *region = nullptr, bool pair_hops = false) {
  const int n = st.size();
  std::vector<int> all;
  if (!region) { all.resize(n); for (int i = 0; i < n; ++i) all[i] = i; region = &all; }
  const std::vector<int> &R = *region;
  const int *q = &(*st.n)[0];
  const FP *v = &(*st.v)[0];
  const FP mu = st.mu, eta = st.eta;
  for (int it = 0; it < cap; ++it) {
    FP best = -tol; int bs[2] = {-1, -1}, bd[2] = {0, 0}, bm = 0;
    for (int i : R) {
      const FP vii = (*st.V)(i, i);
      // d=-1 (gain electron): dE = v + vii/2, chem = +mu - eta*[was +]
      if (q[i] > -1) {
        const FP g = v[i] + 0.5 * vii + mu - (q[i] == 1 ? eta : 0);
        if (g < best) { best = g; bs[0] = i; bd[0] = -1; bm = 1; }
      }
      if (q[i] < 1) { // d=+1 (lose electron): chem = -mu + eta*[becomes +]
        const FP g = -v[i] + 0.5 * vii - mu + (q[i] == 0 ? eta : 0);
        if (g < best) { best = g; bs[0] = i; bd[0] = 1; bm = 1; }
      }
    }
    for (int i : R) {
      if (q[i] >= 1) continue; // donor
      const FP a = -v[i];
      const FP *row = &(st.V->data()[static_cast<std::size_t>(i) * n]);
      for (int j : R) {
        if (j == i || q[j] <= -1 || (!pair_hops && q[j] <= q[i])) continue; // acceptor holds fewer electrons
        const FP e = a + v[j] - row[j];
        if (e < best) { best = e; bs[0] = i; bs[1] = j; bd[0] = 1; bd[1] = -1; bm = 2; }
      }
    }
    if (!bm) return true;
    st.apply(bs, bd, bm, st.dE(bs, bd, bm));
  }
  return false;
}

} // namespace mv
} // namespace phys
#endif
