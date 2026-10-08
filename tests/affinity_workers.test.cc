// Copyright 2026 SiQAD contributors. Licensed under Apache-2.0.
// This separate executable substitutes only the affinity syscall, never the solver.
#define CATCH_CONFIG_MAIN
#include "tests/catch2_wrapper.hpp"
#include <cerrno>
#include <cstring>
#ifdef __linux__
#include <sched.h>
namespace {
enum class Scenario { LargeMask, Denied, TooLarge, Empty };
Scenario scenario = Scenario::LargeMask;
unsigned calls = 0;
int test_getaffinity(pid_t, std::size_t bytes, cpu_set_t *mask) {
  ++calls;
  if (scenario == Scenario::Denied) { errno = EPERM; return -1; }
  if (scenario == Scenario::TooLarge || bytes < 512) {
    errno = EINVAL; return -1;
  }
  std::memset(mask, 0, bytes);
  if (scenario != Scenario::Empty) {
    auto words = reinterpret_cast<unsigned long *>(mask);
    for (unsigned cpu : {0u, 1111u, 2048u})
      words[cpu / (8 * sizeof(unsigned long))] |=
          1ul << (cpu % (8 * sizeof(unsigned long)));
  }
  return 0;
}
}
#define sched_getaffinity test_getaffinity
#endif
#include "src/affinity_workers.h"
#ifdef __linux__
#undef sched_getaffinity

TEST_CASE("Automatic workers respect a sparse affinity mask beyond CPU_SETSIZE") {
  scenario = Scenario::LargeMask; calls = 0;
  REQUIRE(simanneal_affinity::affinityCpuCount() == 3);
  REQUIRE(calls == 3);
  REQUIRE(simanneal_affinity::workerCount(0, 128) == 3);
  REQUIRE(simanneal_affinity::workerCount(0, 2) == 2);
  calls = 0;
  REQUIRE(simanneal_affinity::workerCount(64, 128) == 64);
  REQUIRE(simanneal_affinity::workerCount(128, 32) == 32);
  REQUIRE(calls == 0);
}

TEST_CASE("Unavailable or oversized affinity masks use the bounded hardware fallback") {
  const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
  for (auto failure : {Scenario::Denied, Scenario::TooLarge, Scenario::Empty}) {
    scenario = failure; calls = 0;
    REQUIRE(simanneal_affinity::affinityCpuCount() == 0);
    REQUIRE(calls == (failure == Scenario::Denied ? 1u :
                     failure == Scenario::TooLarge ? 14u : 3u));
    REQUIRE(simanneal_affinity::autoCpuCount() == hardware);
    REQUIRE(simanneal_affinity::workerCount(0, 1) == 1);
  }
}
#else
TEST_CASE("Platforms without Linux affinity retain hardware-based automatic workers") {
  REQUIRE(simanneal_affinity::affinityCpuCount() == 0);
  REQUIRE(simanneal_affinity::autoCpuCount() ==
          std::max(1u, std::thread::hardware_concurrency()));
  REQUIRE(simanneal_affinity::workerCount(0, 1) == 1);
  REQUIRE(simanneal_affinity::workerCount(64, 128) == 64);
  REQUIRE(simanneal_affinity::workerCount(128, 32) == 32);
}
#endif

TEST_CASE("Geometry workers honor threshold, debug logging and explicit limits") {
#ifdef __linux__
  scenario = Scenario::LargeMask;
#endif
  const unsigned available=simanneal_affinity::autoCpuCount();
  const unsigned automatic=std::min(8u,available);
  REQUIRE(simanneal_affinity::geometryWorkerCount(0,511,false)==1);
  REQUIRE(simanneal_affinity::geometryWorkerCount(64,512,true)==1);
  REQUIRE(simanneal_affinity::geometryWorkerCount(0,512,false)==automatic);
  REQUIRE(simanneal_affinity::geometryWorkerCount(1,651,false)==1);
  REQUIRE(simanneal_affinity::geometryWorkerCount(2,651,false)==std::min(2u,automatic));
  REQUIRE(simanneal_affinity::geometryWorkerCount(64,651,false)==automatic);
}
