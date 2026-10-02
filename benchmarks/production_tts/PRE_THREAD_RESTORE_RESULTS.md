> Historical cohort before caller-thread restoration. The following final replay uses the same seeds, so the repeated observations are not independent samples.

# Production qualification after review fixes

The fresh cohort measured **61.9% lower summed estimated TTS997** for the combined policy against optimized portable fixed refinement on the ten large initial targets. The geometric-mean reduction was **30.5%**, with improvements on **7/10 finite pairs**. These compare complete solver jobs, not isolated kernel timings.

## Fresh measured comparisons

Each cell contains 64 jobs. Lower values are better. The sum combines per-layout TTS estimates, not elapsed campaign time.

| Comparison | Summed TTS997 or finite coverage | Reduction |
|---|---:|---:|
| Optimized portable fixed → combined, ten large initial targets | 16.0199 s → 6.0979 s | 61.9% |
| Optimized portable fixed → fixed OpenBLAS, ten large initial targets | 16.0199 s → 15.0635 s | 6.0% |
| Integrated legacy → combined, three exact controls | 2/3 finite pairs | No full-corpus estimate |

The 26-site HA control had 0/64 legacy hits and 14/64 combined-policy hits. The earlier 91.1% exact-control result remains historical and is not repeated as a fresh full-corpus estimate.

The optimized portable-to-OpenBLAS comparison regressed by 4.7% on the three small controls: summed TTS rose from 27.928 ms to 29.243 ms. Three large initial-target estimates also regressed: parity generator by 0.9%, c17 by 0.5%, and xor5Maj by 7.1%. Keep portable and fixed refinement selectable.

The longer legacy schedule produced finite initial-target comparisons on 5/10 large layouts. Its descriptive common-finite reduction was 99.0%. Zero-hit cells prevent a full-corpus claim. Midpoint targets have 8/10 finite pairs, and strongest targets have 3/10. Component percentages overlap and must not be added.

Large layouts contain 103–448 sites. Their frozen witnesses are not ground-state certificates. Exact controls contain 26, 29, and 30 sites. Sparse hits create substantial uncertainty; [analysis.json](results_review/analysis.json) retains all counts, zero-hit cells, and Wilson success-probability intervals. Those are not TTS confidence intervals.

## Review changes and configuration

Shared refinement now chooses a portable, unbiased nonzero offset strictly between fixed centers. Layouts with fewer than 64 sites intentionally use a single-cache fallback. Metadata reports actual nonempty geometry count and fallback status. This changes seeded shared trajectories, so this report uses a fresh complete cohort. The [pre-review results](PRE_REVIEW_RESULTS.md) and their original files remain available.

The five arms, physical model, frozen targets, candidate budgets, and schedules remain unchanged. The preselected combined policy uses shared K10 on nine large layouts and fixed K6 on the remaining large layout and three exact controls. Optimized arms explicitly enable the transient domain mask. Legacy remains the production default; refinement and transient masking remain opt-in.

The host was a Ryzen 7 5800X3D with 16 annealing workers. The supported Ubuntu OpenBLAS 0.3.20 library used one BLAS thread. GCC 11.4 used Release optimization, strict floating-point operations, and native host instructions. The fresh campaign contains 4,160 jobs. Qualification used a separate seed range. [plan.json](results_review/plan.json) records the binary, fixture hash, and seed plan.

Timing includes parameter and model allocation, geometry construction, annealing, repair, refinement, full result checks, and owned cleanup. It excludes process startup, request parsing, and stdout. The formula in [README.md](README.md) has a one-job floor and leaves zero-hit TTS undefined. Both legacy comparators use the integrated corrected validator and lifecycle, not untouched master.

## Audit and tests

The independent implementation audit checked 1,468 unique states from 3,593 claimed-valid rows and rejected none. Maximum energy disagreement was 3.2e-14 eV. It reconstructs the solver's model, including the legacy rounded constants. It does not validate the physical accuracy of those constants.

Audit checks, failure counts, campaign deadlines, and output validation remain active under `python -O`. Regression tests cover corrupted and duplicate energies, invalid charge states, process failures, missing rows, wrong IDs, exhausted campaign budgets, and process-group timeouts. The original cohort also passes the strengthened audit under `-O`.

Native C++/CLI, SWIG/Python, and ASAN/UBSAN tests passed. Sanitizer leak detection remains disabled because third-party connector leaks are outside this qualification. Four Linux CI configurations cover Ubuntu 22.04/24.04 and OpenBLAS OFF/ON. Tests also cover portable seeded offsets and single-cache metadata.

[measurement.json](results_review/measurement.json) and [audit.json](results_review/audit.json) record source and raw-row identity. Raw evidence, binaries, source snapshots, and dependency provenance are saved in a new ignored archive. Earlier archives are unchanged. See [attribution](../../docs/ATTRIBUTION.md) for borrowed methods and licenses.
