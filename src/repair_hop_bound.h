// Conservative exact hop selection, adapted from qualified SimAnneal research.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
namespace simanneal_pair_bound {
struct Geometry {
  int size;
  double tau;
  std::vector<std::vector<int>> strong;
  template <class Matrix>
  Geometry(int count, const Matrix &matrix, double threshold = .05)
      : size(count), tau(threshold), strong(count) {
    for (int i = 0; i < count; ++i)
      for (int j = 0; j < count; ++j)
        if (matrix(i, j) > tau)
          strong[i].push_back(j);
  }
};
struct Workspace {
  std::array<std::vector<int>, 3> targets;
};
struct Move {
  int from = -1, to = -1;
  double delta;
  explicit Move(double eps) : delta(-eps) {}
};
template <class Charges, class Potentials, class Matrix>
Move select(const Geometry &geometry, Workspace &work, const Charges &charge,
            const Potentials &potential, const Matrix &matrix, double eps) {
  Move best(eps);
  bool finite = std::isfinite(geometry.tau);
  for (int i = 0; i < geometry.size; ++i)
    finite &= std::isfinite(potential[i]);
  if (!finite) {
    // std::sort cannot use the ordinary numerical comparator with NaNs.
    // Reproduce the original nested scan, including its nonfinite comparisons.
    for (int i = 0; i < geometry.size; ++i)
      for (int j = 0; j < geometry.size; ++j)
        if (charge[i] < charge[j]) {
          const double delta = -potential[i] + potential[j] - matrix(i, j);
          if (delta < best.delta) {
            best.delta = delta;
            best.from = i;
            best.to = j;
          }
        }
    return best;
  }
  const auto examine = [&](int i, int j) {
    const double delta = -potential[i] + potential[j] - matrix(i, j);
    const bool earlier =
        best.from < 0 || i < best.from || (i == best.from && j < best.to);
    if (delta < -eps &&
        (delta < best.delta || (delta == best.delta && earlier))) {
      best.delta = delta;
      best.from = i;
      best.to = j;
    }
  };
  // Visit every strong pair before any weak bound is used.
  for (int i = 0; i < geometry.size; ++i)
    for (const int j : geometry.strong[i])
      if (charge[i] < charge[j])
        examine(i, j);
  for (auto &list : work.targets)
    list.clear();
  for (int j = 0; j < geometry.size; ++j)
    work.targets[charge[j] + 1].push_back(j);
  for (auto &list : work.targets)
    std::sort(list.begin(), list.end(), [&](int a, int b) {
      return potential[a] < potential[b] ||
             (potential[a] == potential[b] && a < b);
    });
  for (int i = 0; i < geometry.size; ++i)
    for (int cls = charge[i] + 2; cls < 3; ++cls)
      for (const int j : work.targets[cls]) {
        const double lower = -potential[i] + potential[j] - geometry.tau;
        // Covers subtraction and comparison rounding; overflow/nonfinite
        // margins disable pruning. Strictly beyond best retains all ties and
        // boundaries.
        const double scale = 1 + std::abs(potential[i]) +
                             std::abs(potential[j]) + std::abs(geometry.tau) +
                             std::abs(best.delta);
        const double margin =
            16 * std::numeric_limits<double>::epsilon() * scale;
        if (std::isfinite(lower) && std::isfinite(margin) &&
            lower > best.delta + margin)
          break;
        if (matrix(i, j) <= geometry.tau)
          examine(i, j);
      }
  return best;
}
} // namespace simanneal_pair_bound
