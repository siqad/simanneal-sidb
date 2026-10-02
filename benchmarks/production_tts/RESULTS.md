# Production qualification: 2026-10-02

The combined policy reduced summed estimated TTS997 by **59.0%** against optimized portable fixed refinement on ten large layouts. The geometric-mean reduction was **29.5%**, with improvements on eight layouts. This comparison isolates the additional OpenBLAS/shared-strategy benefit after the optimized profile and fixed refinement are already enabled.

The complete stack reduced summed estimated TTS997 by **91.1%** against the integrated legacy profile on three exact controls. Large-layout legacy runs had too few hits to support a corpus-wide percentage. There is no measured universal speedup against untouched master.

## Measured comparisons

Each cell has 64 complete jobs. Lower values are better. The sum is the sum of each layout's TTS estimate, not the elapsed benchmark time.

| Comparison | Targets and coverage | Before → after, summed TTS997 | Reduction |
|---|---|---:|---:|
| Optimized portable fixed → combined | Initial witnesses, all 10 large layouts | 15.651 s → 6.417 s | 59.0% |
| Optimized portable fixed → fixed OpenBLAS | Initial witnesses, all 10 large layouts | 15.651 s → 14.488 s | 7.4% |
| Fixed OpenBLAS → combined shared/fixed policy | Initial witnesses, all 10 large layouts | 14.488 s → 6.417 s | 55.7% |
| Integrated legacy → combined | Exact ground states, all 3 controls | 368.719 ms → 32.809 ms | 91.1% |
| Longer legacy schedule → combined | Initial witnesses, only 4 of 10 large layouts have finite estimates in both arms | 100.530 s → 1.225 s | 98.8%, descriptive subset only |

These percentages overlap and must not be added. The combined policy uses shared refinement on the nine K10 layouts and fixed refinement on K6 layouts. This policy was selected before this cohort. K6 runs in the nominal shared arm use the same fixed policy, so their differences reflect timing noise.

The largest contribution to the initial-target summed reduction comes from the parity-check layout: hits increased from 3/64 with portable fixed refinement to 10/64 with shared OpenBLAS. Its estimated TTS fell from 10.366 s to 2.936 s. Sparse hits make these estimates uncertain. The FA and c17 initial-target estimates regressed by approximately 0.8% and 5.7%, respectively. Both already hit their targets in every job; extra refinement added cost.

For midpoint targets, only 8/10 large layouts had finite estimates in both optimized arms. For the strongest targets, only 4/10 did. No full-corpus reduction is claimed for those thresholds. The longer legacy schedule had initial-target hits on only 4/10 large layouts. The ordinary legacy schedule had hits on only 1/10.

The exact controls contain 26, 29, and 30 sites. Optimized K6 had 12/64, 64/64, and 64/64 exact hits, respectively. Legacy had 2/64, 3/64, and 3/64. The 26-site control still has substantial sampling uncertainty.

## Configuration and timing

The host was an AMD Ryzen 7 5800X3D with 16 annealing workers. GCC 11.4 used Release optimization, strict floating-point operations, and native host instructions. OpenBLAS was the supported Ubuntu 0.3.20 pthread LP64 library, with one BLAS thread. The campaign completed in 410 seconds. Host observations before and after showed no competing CPU or GPU workload.

The corpus has ten large layouts with 103–448 sites and three exact controls. The five arms produced 4,160 jobs. Matched seed intervals were separate from qualification and prior research. All original schedules, restart counts, physics, and thresholds are in `cases.json`. The longer legacy schedule uses at least 512 cycles and hop factor 5 on large layouts.

Optimized arms enable PCG32, bounded repair, the singleton shortcut, finite-grid probability shortcuts, and the transient domain mask. They refine up to eight candidates with one or four rounds, as frozen per layout. K6 and K10 retain their explicit settings. The production default remains legacy; the transient mask and refinement are not enabled by default.

The timer includes parameter/model allocation, geometry construction, annealing, repair, refinement, full result validation, and owned cleanup. It excludes process startup, request parsing, and stdout. TTS997 uses observed complete-job success and the formula in [README.md](README.md), with a one-job floor and undefined zero-hit cells.

Both legacy comparators use this PR's corrected validator and lifecycle. Neither is an untouched-master binary. The large thresholds are frozen validated witnesses, not ground-state certificates. Reused layouts and schedules make this an integration qualification, not an unseen-layout generalization study.

## Validation and reproducibility

The post-timing independent audit checked 1,485 unique configurations among 3,592 valid outputs. It rejected none. Maximum energy disagreement was 3.91e-14 eV. Duplicate outputs also had their reported energies checked. A separate arithmetic check reproduced every cell's mean, hit count, TTS, phase sum, and matched seeds.

[analysis.json](results/analysis.json) contains every cell, target, Wilson 95% success-probability interval, and explicit zero-hit exclusions. These intervals are not TTS confidence intervals. [measurement.json](results/measurement.json), [plan.json](results/plan.json), and [audit.json](results/audit.json) identify the source, binary, fixtures, and raw-row hashes. Raw inputs, outputs, binaries, logs, and host observations are retained in the local evidence archive.

Local macOS Release, portable ASAN/UBSAN, native CLI, and SWIG/Python tests passed. Leak detection was disabled for sanitizer tests because third-party connector leaks are outside this qualification. Ubuntu 22.04 and 24.04 CI passed with OpenBLAS both enabled and disabled, including a Python-wrapper job.

A separate seven-site native CLI control included process startup and XML output. It observed 60/64 legacy hits versus 64/64 optimized K6 hits, with estimated TTS997 of 15.29 ms versus 7.22 ms. This small control does not establish large-layout CLI performance. An earlier overlapping-build pilot was excluded.

## PR scope

Keep optimized search, fixed/shared refinement, transient masking, and numerical backends selectable. Shared refinement is useful but does not win on every target. Preserve the legacy default while this reduced qualification remains the evidence base. CUDA, broader policy tuning, and a SiQAD submodule update are separate work.

See [attribution](../../docs/ATTRIBUTION.md) for QuickExact, ClusterComplete, fiction, PCG, Lemire, and OpenBLAS credit.
