#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
// Independent adaptation of QuickExact charge bounds: Drewniok, Walter, Wille,
// ASP-DAC 2024, arXiv:2308.04487. Signs/tolerances follow SimAnneal's model.
namespace simanneal_domains {
template <class Matrix, class Fields>
std::vector<unsigned char> finalCharges(const Matrix &a, const Fields &external,
                                        const Fields &fixed, int n, double mu,
                                        double eta, double tolerance) {
  std::vector<unsigned char> domains(n, 7);
  for (int i = 0; i < n; ++i) {
    const double field = external[i] + fixed[i];
    double lower = -field, upper = -field, scale = std::abs(field);
    bool finite = std::isfinite(field);
    for (int j = 0; j < n && finite; ++j)
      if (i != j) {
        const double coupling = a(i, j);
        finite = std::isfinite(coupling) && coupling >= 0;
        lower -= coupling;
        upper += coupling;
        scale += std::abs(coupling);
      }
    if (!finite)
      continue;
    const double margin = tolerance + 1e-10 * (1 + scale) * n;
    unsigned char mask = 7;
    if (upper + mu - eta < -margin)
      mask &= ~4;
    if (lower + mu > margin)
      mask &= ~1;
    if (upper + mu < -margin || lower + mu - eta > margin)
      mask &= ~2;
    if (mask)
      domains[i] = mask; // Empty intersections never manufacture a charge.
  }
  return domains;
}
inline bool singleton(unsigned char mask) {
  return mask == 1 || mask == 2 || mask == 4;
}
inline int singletonCharge(unsigned char mask) {
  return mask == 1 ? -1 : mask == 2 ? 0 : 1;
}
} // namespace simanneal_domains
