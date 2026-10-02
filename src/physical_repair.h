#pragma once
#include "global.h"
#include "repair_hop_bound.h"
#include <boost/numeric/ublas/vector.hpp>
namespace phys {
struct SimParams;
struct RepairResult {
  boost::numeric::ublas::vector<int> config;
  FPType energy = std::numeric_limits<FPType>::infinity();
  bool valid = false, budget_exhausted = false;
  int hops = 0, population_changes = 0, repair_passes = 0;
};
// Called with the active immutable model. Every successful return is fully
// checked.
RepairResult boundedPhysicalRepair(const SimParams &,
                                   const boost::numeric::ublas::vector<int> &,
                                   const simanneal_pair_bound::Geometry *,
                                   bool known_initial_invalid = false);
} // namespace phys
