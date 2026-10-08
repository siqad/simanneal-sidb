// Copyright 2026 SiQAD contributors. Licensed under Apache-2.0.
#ifndef SIMANNEAL_AFFINITY_WORKERS_H
#define SIMANNEAL_AFFINITY_WORKERS_H

#include <algorithm>
#include <new>
#include <thread>
#ifdef __linux__
#include <cerrno>
#include <sched.h>
#include <vector>
#endif

namespace simanneal_affinity {

// Zero means unavailable. Grow the mask to support CPU IDs above CPU_SETSIZE.
inline unsigned affinityCpuCount() {
#ifdef __linux__
  const std::size_t max_bytes = 1024 * 1024;
  for (std::size_t bytes = 128; bytes <= max_bytes; bytes *= 2) {
    std::vector<unsigned long> words;
    try { words.assign(bytes / sizeof(unsigned long), 0); }
    catch (const std::bad_alloc &) { return 0; }
    if (::sched_getaffinity(0, bytes,
            reinterpret_cast<cpu_set_t *>(words.data())) == 0) {
      unsigned count = 0;
      for (unsigned long word : words) {
        while (word) { word &= word - 1; ++count; }
      }
      return count;
    }
    if (errno != EINVAL) return 0;
  }
#endif
  return 0;
}

inline unsigned autoCpuCount() {
  const unsigned available = affinityCpuCount();
  return available ? available : std::max(1u, std::thread::hardware_concurrency());
}

// Initialization validates requested >= 0 and restarts > 0 before this call.
inline int workerCount(int requested, int restarts) {
  if (requested) return std::min(requested, restarts);
  return static_cast<int>(std::min(autoCpuCount(), static_cast<unsigned>(restarts)));
}

// Small matrices and debug output stay serial. Preparation never exceeds the
// caller's worker limit, the current affinity mask, or eight threads.
inline int geometryWorkerCount(int requested, int sites, bool debug_logging) {
  if (sites < 512 || debug_logging || requested == 1) return 1;
  const unsigned limit = requested > 0 ? std::min(requested, 8) : 8;
  return static_cast<int>(std::min(autoCpuCount(), limit));
}

} // namespace simanneal_affinity
#endif
