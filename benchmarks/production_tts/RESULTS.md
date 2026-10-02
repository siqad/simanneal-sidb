# Final production qualification after thread restoration

The final same-seed replay measured **62.3% lower summed estimated TTS997** for the combined policy against optimized portable fixed refinement on the ten large initial targets. The geometric-mean reduction was **30.8%**, with improvements on **9/10 finite pairs**. These compare complete solver jobs, not isolated kernel timings.

All 4,160 per-job validity, configuration, and energy outputs matched the preceding cohort exactly. The preceding summed reduction was 61.9%; the replay measured 62.3%. These are repeated seeds, not independent success samples. Runtime variation does not isolate the cost of thread restoration.

## Final measured comparisons

Each cell contains 64 jobs. Lower values are better. The sum combines per-layout TTS estimates, not elapsed campaign time.

| Comparison | Summed TTS997 or finite coverage | Reduction |
|---|---:|---:|
| Optimized portable fixed → combined, ten large initial targets | 16.6042 s → 6.2521 s | 62.3% |
| Optimized portable fixed → fixed OpenBLAS, ten large initial targets | 16.6042 s → 15.2061 s | 8.4% |
| Integrated legacy → combined, three exact controls | 2/3 finite pairs | No full-corpus estimate |

The 26-site control again had 0/64 legacy hits and 14/64 optimized hits, so no full three-control legacy reduction is reported. OpenBLAS regressed by 3.7% on the small optimized controls: summed TTS rose from 27.001 ms to 27.989 ms. Both portable and fixed strategies remain available.

The longer legacy schedule produced finite initial-target comparisons on 5/10 large layouts. Its descriptive common-finite reduction was 99.0%. Zero-hit cells prevent a full-corpus claim. Midpoint targets have 8/10 finite pairs, and strongest targets have 3/10. Component percentages overlap and must not be added.

Large layouts contain 103–448 sites. Their frozen witnesses are not ground-state certificates. Exact controls contain 26, 29, and 30 sites. Sparse hits create substantial uncertainty; [analysis.json](results_final/analysis.json) retains all counts, zero-hit cells, and Wilson success-probability intervals. Those are not TTS confidence intervals.

## Review changes and configuration

OpenBLAS thread control is scoped to the search. Construction preserves the caller setting. Normal return, exception unwind, and singleton return restore it. Matching counts avoid setter calls. The benchmark records the count inside the search.

Shared refinement now chooses a portable, unbiased nonzero offset strictly between fixed centers. Layouts with fewer than 64 sites intentionally use a single-cache fallback. Metadata reports actual nonempty geometry count and fallback status. The preceding review cohort changed shared trajectories. This report replays that cohort after adding scoped OpenBLAS thread restoration, without pooling the repeated seeds as independent samples. The [preceding review results](PRE_THREAD_RESTORE_RESULTS.md) and their original files remain available.

The five arms, physical model, frozen targets, candidate budgets, and schedules remain unchanged. The preselected combined policy uses shared K10 on nine large layouts and fixed K6 on the remaining large layout and three exact controls. Optimized arms explicitly enable the transient domain mask. Legacy remains the production default; refinement and transient masking remain opt-in.

The host was a Ryzen 7 5800X3D with 16 annealing workers. The supported Ubuntu OpenBLAS 0.3.20 library used one BLAS thread. GCC 11.4 used Release optimization, strict floating-point operations, and native host instructions. The replay contains 4,160 jobs. Its explicit replay metadata references the preceding raw-row hash. Qualification replays its separate seed range. [plan.json](results_final/plan.json) records the binary, fixture hash, and seed plan.

Timing includes parameter and model allocation, geometry construction, annealing, repair, refinement, full result checks, and owned cleanup. It excludes process startup, request parsing, and stdout. The formula in [README.md](README.md) has a one-job floor and leaves zero-hit TTS undefined. Both legacy comparators use the integrated corrected validator and lifecycle, not untouched master.

## Audit and tests

The independent implementation audit checked 1,468 unique states from 3,593 claimed-valid rows and rejected none. Maximum energy disagreement was 3.2e-14 eV. It reconstructs the solver's model, including the legacy rounded constants. It does not validate the physical accuracy of those constants.

Audit checks, failure counts, campaign deadlines, and output validation remain active under `python -O`. Regression tests cover corrupted and duplicate energies, invalid charge states, process failures, missing rows, wrong IDs, exhausted campaign budgets, and process-group timeouts. The original cohort also passes the strengthened audit under `-O`.

Native C++/CLI, SWIG/Python, and ASAN/UBSAN tests passed. Sanitizer leak detection remains disabled because third-party connector leaks are outside this qualification. Four Linux CI configurations cover Ubuntu 22.04/24.04 and OpenBLAS OFF/ON. Tests also cover portable seeded offsets and single-cache metadata.

[measurement.json](results_final/measurement.json) and [audit.json](results_final/audit.json) record source and raw-row identity. Raw evidence, binaries, source snapshots, and dependency provenance are saved in a new ignored archive. Earlier archives are unchanged. See [attribution](../../docs/ATTRIBUTION.md) for borrowed methods and licenses.

## Subsequent API and metadata guards

The final merge also rejects reuse of refinement geometry with a changed byte cap. Solver jobs construct and execute geometry with the same cap, so this does not alter the measured search. In OpenBLAS-enabled builds, unused BLAS backends now report zero active BLAS threads. The archived replay predates this metadata correction and reports the process setting even for portable arms. Backend identity remains explicit in each row. The timings above belong to the recorded binary; these two guards received regression tests without another timing campaign.
