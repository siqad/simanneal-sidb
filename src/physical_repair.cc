#include "physical_repair.h"
#include "repair_class_scan.h"
#include "simanneal.h"
namespace phys {
RepairResult
boundedPhysicalRepair(const SimParams &sp, const ublas::vector<int> &initial,
                      const simanneal_pair_bound::Geometry *geometry,
                      bool known_initial_invalid) {
  RepairResult out;
  if (initial.size() != static_cast<std::size_t>(sp.n_dbs) ||
      !sp.population_finite_matrix)
    return out;
  for (int q : initial)
    if (q < -1 || q > 1)
      return out;
  auto n = initial;
  ublas::vector<FPType> v(sp.n_dbs);
  for (int i = 0; i < sp.n_dbs; ++i) {
    if (!std::isfinite(sp.v_ext[i]) || !std::isfinite(sp.v_fc[i]))
      return out;
    v[i] = -(sp.v_ext[i] + sp.v_fc[i]);
    for (int j = 0; j < sp.n_dbs; ++j)
      v[i] -= sp.v_ij(i, j) * n[j];
  }
  const double eps = constants::RECALC_STABILITY_ERR;
  const auto site_valid = [&](int i) {
    return (n[i] == -1 && v[i] + sp.mu < eps) ||
           (n[i] == 1 && v[i] + sp.mu - constants::eta > -eps) ||
           (n[i] == 0 && v[i] + sp.mu > -eps &&
            v[i] + sp.mu - constants::eta < eps);
  };
  const auto change = [&](int i, int delta) {
    n[i] += delta;
    const FPType *row = &sp.v_ij.data()[static_cast<std::size_t>(i) * sp.n_dbs];
    for (int k = 0; k < sp.n_dbs; ++k)
      v[k] -= row[k] * delta;
  };
  FPType energy;
  if (!known_initial_invalid && SimAnneal::validatedEnergy(initial, energy)) {
    out.config = initial;
    out.energy = energy;
    out.valid = true;
  }
  const int maximum_hops = 4 * sp.n_dbs;
  simanneal_pair_bound::Workspace bound_workspace;
  simanneal_class_scan::Workspace class_workspace;
  for (int rounds = 0; rounds < maximum_hops + 8; ++rounds) {
    bool pop_valid = true;
    for (int i = 0; i < sp.n_dbs; ++i)
      pop_valid &= site_valid(i);
    if (!pop_valid && out.repair_passes < 8) {
      ++out.repair_passes;
      for (int i = 0; i < sp.n_dbs; ++i)
        if (!site_valid(i)) {
          const int target = v[i] + sp.mu < 0                    ? -1
                             : v[i] + sp.mu - constants::eta > 0 ? 1
                                                                 : 0;
          if (target != n[i]) {
            change(i, target - n[i]);
            ++out.population_changes;
          }
        }
      pop_valid = true;
      for (int i = 0; i < sp.n_dbs; ++i)
        pop_valid &= site_valid(i);
    }
    int from = -1, to = -1;
    if (geometry && sp.n_dbs >= 256) {
      const auto move = simanneal_pair_bound::select(*geometry, bound_workspace,
                                                     n, v, sp.v_ij, eps);
      from = move.from;
      to = move.to;
    } else {
      const auto move =
          simanneal_class_scan::select(class_workspace, n, v, sp.v_ij, eps);
      from = move.from;
      to = move.to;
    }
    if (from < 0) {
      if (pop_valid && SimAnneal::validatedEnergy(n, energy)) {
        if (!out.valid || energy < out.energy) {
          out.config = n;
          out.energy = energy;
          out.valid = true;
        }
        return out;
      }
      if (out.repair_passes >= 8) {
        out.budget_exhausted = true;
        return out;
      }
      continue;
    }
    if (out.hops >= maximum_hops) {
      out.budget_exhausted = true;
      return out;
    }
    change(from, 1);
    change(to, -1);
    ++out.hops;
  }
  out.budget_exhausted = true;
  return out; // Any retained incumbent was validated against the original full
              // model.
}
} // namespace phys
