#ifndef SIMANNEAL_HOP_SELECTOR_H
#define SIMANNEAL_HOP_SELECTOR_H

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace phys {
// Optimizer proposal policies, not physical tunnelling rates.
enum HopSelection { UniformHop, LocalUniformHop, LocalDistanceHop, LocalRadiusHop };

// Per-worker reverse-indexed neutral-neighbor lists for radius proposals.
struct RadiusEligibleCache {
  bool initialized = false;
  std::vector<int> eligible;
  std::vector<int> inverse;
  std::vector<int> counts;

  void reset(std::size_t sites, std::size_t edges) {
    eligible.resize(edges);
    inverse.assign(edges, -1);
    counts.assign(sites, 0);
    initialized = false;
  }
};

// Immutable geometry shared by all workers. No per-proposal allocation or exp.
class HopNeighborhood {
public:
  struct Neighbor { int site; double weight; };

  template<class Distances>
  void build(const Distances &distances, int size, int requested_k,
             double length_nm, bool weighted) {
    if (requested_k < 1 || !std::isfinite(length_nm) || length_nm <= 0)
      throw std::invalid_argument("hop_neighbors and hop_length_nm must be positive");
    count = std::min(requested_k, std::max(0, size - 1));
    rows.clear();
    rows.resize(static_cast<std::size_t>(size) * count);
    std::vector<int> candidates;
    candidates.reserve(size);
    for (int i = 0; i < size; ++i) {
      candidates.clear();
      for (int j = 0; j < size; ++j) if (i != j) candidates.push_back(j);
      auto closer = [&](int a, int b) {
        return distances(i,a) == distances(i,b) ? a < b : distances(i,a) < distances(i,b);
      };
      if (count < static_cast<int>(candidates.size()))
        std::nth_element(candidates.begin(), candidates.begin() + count, candidates.end(), closer);
      std::sort(candidates.begin(), candidates.begin() + count, closer);
      const double nearest = count ? distances(i,candidates[0]) : 0;
      for (int j = 0; j < count; ++j) {
        const int site = candidates[j];
        // A common row scale cancels on normalization; subtracting the nearest
        // distance avoids all weights underflowing on widely separated layouts.
        rows[static_cast<std::size_t>(i) * count + j] = {site, weighted ?
          std::exp(-(distances(i,site) - nearest) / (length_nm * 1e-9)) : 1.0};
      }
    }
  }

  template<class Distances>
  void buildRadius(const Distances &distances, int size, double radius_nm) {
    if (size < 0 || !std::isfinite(radius_nm) || radius_nm <= 0)
      throw std::invalid_argument("hop_radius_nm must be finite and positive");
    const double radius = radius_nm * 1e-9;
    offsets.assign(static_cast<std::size_t>(size) + 1, 0);
    neighbors.clear();
    reverse.assign(static_cast<std::size_t>(size), {});
    max_degree = 0;

    for (int source = 0; source < size; ++source) {
      for (int site = 0; site < size; ++site) {
        const double distance = distances(source, site);
        if (source != site && std::isfinite(distance) && distance >= 0
            && distance <= radius) {
          const int edge = static_cast<int>(neighbors.size());
          neighbors.push_back(site);
          reverse[site].push_back({source, edge});
        }
      }
      offsets[source + 1] = static_cast<int>(neighbors.size());
      max_degree = std::max(max_degree, offsets[source + 1] - offsets[source]);
    }
  }

  template<class Charges>
  int select(int source, const Charges &charges, double unit_random) const {
    const std::size_t start = static_cast<std::size_t>(source) * count;
    double total = 0;
    for (int j = 0; j < count; ++j) {
      const auto &entry = rows[start+j];
      if (charges[entry.site] == 0) total += entry.weight;
    }
    if (total <= 0) return -1; // caller falls back to a global neutral target
    double threshold = unit_random * total;
    int last = -1;
    for (int j = 0; j < count; ++j) {
      const auto &entry = rows[start+j];
      if (charges[entry.site] != 0 || entry.weight <= 0) continue;
      last = entry.site;
      threshold -= entry.weight;
      if (threshold < 0) return last;
    }
    return last; // rounding at the upper endpoint
  }

  template<class Charges>
  int selectRadius(int source, const Charges &charges, double unit_random,
                   RadiusEligibleCache &cache) const {
    if (!cache.initialized) {
      cache.reset(reverse.size(), neighbors.size());
      for (std::size_t site = 0; site < reverse.size(); ++site)
        if (charges[site] == 0) setNeutral(static_cast<int>(site), true, cache);
      cache.initialized = true;
    }
    const int count = cache.counts[source];
    if (count == 0) return -2; // strict radius: consume a null attempt
    const int slot = std::min(count - 1, static_cast<int>(unit_random * count));
    const int edge = cache.eligible[offsets[source] + slot];
    return neighbors[edge];
  }

  void setNeutral(int site, bool neutral, RadiusEligibleCache &cache) const {
    for (const auto &entry : reverse[site]) {
      const int source = entry.source;
      const int edge = entry.edge;
      const int slot = cache.inverse[edge];
      if (neutral) {
        if (slot >= 0) continue;
        const int position = cache.counts[source]++;
        cache.eligible[offsets[source] + position] = edge;
        cache.inverse[edge] = position;
      } else {
        if (slot < 0) continue;
        const int last = --cache.counts[source];
        const int moved = cache.eligible[offsets[source] + last];
        cache.eligible[offsets[source] + slot] = moved;
        cache.inverse[moved] = slot;
        cache.inverse[edge] = -1;
      }
    }
  }

  int neighborsPerSite() const { return max_degree ? max_degree : count; }
  std::size_t edgeCount() const { return neighbors.size(); }
  std::size_t storageBytes() const {
    std::size_t bytes = rows.size() * sizeof(Neighbor)
        + (offsets.size() + neighbors.size()) * sizeof(int)
        + reverse.size() * sizeof(std::vector<ReverseEdge>);
    for (const auto &row : reverse) bytes += row.size() * sizeof(ReverseEdge);
    return bytes;
  }
private:
  struct ReverseEdge { int source; int edge; };
  int count = 0;
  int max_degree = 0;
  std::vector<Neighbor> rows;
  std::vector<int> offsets, neighbors;
  std::vector<std::vector<ReverseEdge>> reverse;
};
}
#endif
