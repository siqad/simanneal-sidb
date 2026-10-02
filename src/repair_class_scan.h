// Charge-class scan preserves the original ascending-pair energy/tie rule.
#pragma once
#include <array>
#include <vector>
namespace simanneal_class_scan {
struct Workspace {
  std::array<std::vector<int>, 3> targets;
};
struct Move {
  int from = -1, to = -1;
  double delta;
  explicit Move(double eps) : delta(-eps) {}
};
template <class Charges, class Potentials, class Matrix>
Move select(Workspace &work, const Charges &charge, const Potentials &potential,
            const Matrix &matrix, double eps) {
  Move best(eps);
  for (auto &list : work.targets)
    list.clear();
  for (int j = 0; j < static_cast<int>(charge.size()); ++j)
    work.targets[charge[j] + 1].push_back(j);
  for (int i = 0; i < static_cast<int>(charge.size()); ++i)
    for (int cls = charge[i] + 2; cls < 3; ++cls)
      for (const int j : work.targets[cls]) {
        const double delta = -potential[i] + potential[j] - matrix(i, j);
        const bool earlier =
            best.from < 0 || i < best.from || (i == best.from && j < best.to);
        if (delta < -eps &&
            (delta < best.delta || (delta == best.delta && earlier))) {
          best.delta = delta;
          best.from = i;
          best.to = j;
        }
      }
  return best;
}
} // namespace simanneal_class_scan
