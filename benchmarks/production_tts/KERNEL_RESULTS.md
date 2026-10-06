# Nibi validation-kernel qualification (2026-10-06)

The combined runtime kernels reduce complete-job time to solution (TTS) against master `a1a3ff5` (PR #8).
They cache unchanged population validity, reuse validation scratch, read contiguous rows, restrict hop checks to eligible charge classes, and use a freshly proven private sparse search validator.
The public dense validator and export contracts remain unchanged. Population probability caching remains disabled by default.

| Comparison against PR #8 | Qualified configuration | Portable configuration |
|---|---:|---:|
| New default kernels, LTO off: summed TTS reduction | 16.56% | 3.74% |
| New kernels with optional LTO: summed TTS reduction | 17.68% | 5.20% |
| New kernels with optional LTO: runtime geometric-mean reduction | 17.03% | 3.85% |

The qualified configuration uses OpenBLAS, transient masking, and the previously qualified refinement settings.
The portable configuration uses portable arithmetic without masking or refinement.
TTS comparisons use 13 finite qualified cells and seven common finite portable cells. Six portable cells had zero hits and have undefined TTS.
The larger targets are validated best-known energies, not certified ground states.
These figures are combined measurements; component gains must not be added.

## Workload and method

Nibi used Intel Xeon 6972P processors, 16 allocated CPUs, 16 GiB memory, and `rrg-wolkow_cpu` on `cpubase_bycore_b1`.
The requested allocation identity was group `rrg-wolkow`, RAPI `ewg-710-ab`, owner `ewg-710`.
Builds used GCC 12.3, Release, strict floating-point flags, native instructions disabled, and one OpenBLAS thread per worker.
The final short comparison used 96 fresh jobs per layout and arm, 16 workers, and paired seeds.
There were three certified exact controls with 26–30 sites and ten larger layouts with 103–448 sites.
Schedules ranged from 32 to 512 cycles and 64 to 512 restarts.
A separate duration check used four larger layouts, 10,000 cycles, and eight jobs per cell.
Long-run hit counts are too small for a general TTS claim. Optional-LTO runtime fell by 8.74% qualified and 6.08% portable in that check.

TTS uses a 99.7% target confidence:
`mean_complete_job_time * max(1, log(0.003) / log(1 - hit_probability))`.
Zero-hit cells remain undefined. A certain hit needs one complete job.
Complete time includes construction, search and refinement, result preparation, and destruction.
Independent physical validation rejected no reported results in the research campaign. Expected paired configurations matched.

The promoted runtime matches the frozen `B13-scope` source after normalizing line endings:

- `src/simanneal.cc`: `8fac5a3ea2b0f5ce9bf2f4ec4f9489ee83c79d4c91d41b4593c7a4523faebd54`
- `src/simanneal.h`: `b7526ac0e9b3052904936befdc28da2e98c4655447461fce189d085ede7eaca5`

The full research archive retains plans, raw rows, source snapshots, binaries, physical checks, and checksum manifests.
Its primary-checkout location is the gitignored `build-research/simanneal-optimization/2026-10-06/nibi-twenty-hypotheses` directory.
Fixture provenance and borrowed methods are credited in [ATTRIBUTION.md](../../docs/ATTRIBUTION.md).

## Simulation versus export

The export timer starts after `invokeSimAnneal` and includes `suggestedConfigResults(true)`.
That call deduplicates, validates, recalculates energies, and prepares result records.
The timer also includes the benchmark's minimum-result reduction and statistics copy.
It excludes file serialization and Python wrapper conversion or transfer.

Shares below divide summed component time by summed complete-job time, rather than averaging per-job percentages.
These shares describe the new default kernels with LTO off.

| Workload | Setup | Simulation and refinement | Export | Cleanup |
|---|---:|---:|---:|---:|
| Short, qualified | 18.94% | 76.91% | 3.86% | 0.28% |
| Short, portable | 6.72% | 86.37% | 6.46% | 0.45% |
| 10,000 cycles, qualified | 1.81% | 97.77% | 0.38% | 0.03% |
| 10,000 cycles, portable | 0.38% | 99.22% | 0.37% | 0.03% |

Before PR #8, export occupied about 30–47% of the short workloads and about 5% of the duration workloads.
After PR #8, it occupied 4–8% and about 0.46%, respectively.
Further optimization should prioritize simulation and refinement. Export is now a small fraction of the longer workloads.
