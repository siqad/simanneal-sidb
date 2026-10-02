# Production time-to-solution benchmark

This runner calls the production `SimAnneal` constructor, search, result validation, and destructor. It contains no alternative repair or refinement implementation.

The reduced corpus contains 13 previously studied layouts. Three controls have prior exact references. Ten larger layouts have frozen, physically validated witnesses and preset energy thresholds. Those thresholds are not ground-state certificates. `cases.json` retains the geometry, physical fields, provenance, targets, and schedules.

See [the production qualification](RESULTS.md) for measured gains, regressions, and evidence limits.

## Build and run

Build with supported OpenBLAS development headers and library:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PRODUCTION_BENCHMARK=ON -DSIMANNEAL_ENABLE_OPENBLAS=ON
cmake --build build --parallel
python3 benchmarks/production_tts/run.py build/production_tts build/tts-results --jobs 64 --workers 16
python3 benchmarks/production_tts/audit.py build/tts-results
```

The audit requires NumPy. The timed runner uses only Python's standard library. Use a fresh output directory. Avoid other compute work on the host during timing. Native host instructions require the separate `SIMANNEAL_NATIVE_ARCH=ON` build option.

The five policies are legacy, optimized portable fixed refinement, fixed refinement with OpenBLAS, shared refinement with OpenBLAS, and a longer legacy schedule. The latter uses at least 512 cycles and hop factor 5 on large layouts. Its settings are fixed before this cohort. Exact controls retain their ordinary schedule.

Optimized policies explicitly enable transient domain masks to match the earlier experimental composition. This is an explicit benchmark choice, not the production default. K6 controls use the same fixed policy in both nominal BLAS arms. Differences between those arms measure timing noise, not a shared-geometry benefit.

Each policy receives matched base seeds and restart counts. RNG changes mean trajectories need not match. Every new run should reserve a seed range that does not overlap earlier studies. The default range starts at 3,280,000,003, with separate case and job offsets. A qualification run must use a different explicit seed. For a deliberate same-seed replay, pass `--replay-of` with the prior rows SHA256. The plan then marks the run as a replay rather than an independent seed cohort.

## Timing and statistics

Each trial is a complete job containing all its restarts. Compute timing includes parameter allocation, model/geometry construction, search, repair, refinement, full result checks, and owned cleanup. It excludes file parsing, process startup, and stdout. Per-process wall time is also recorded, separately. These measurements are not native CLI launch-to-XML timing.

For observed job success probability `p`, use:

```
TTS997 = mean complete-job time * max(1, log(0.003) / log(1 - p))
```

A target hit requires a fully valid output with energy no greater than the frozen threshold plus `1e-8` eV. Zero-hit TTS remains undefined. All-hit TTS has a one-job floor. The runner never adds pseudocounts or drops zero-hit cases from an aggregate without reporting the exclusion.

Each subprocess has a 120-second process-group timeout. The campaign has a 1,800-second cap. Raw requests, outputs, stderr, process timing, binary hash, and fixture hash remain in the output directory. The independent implementation audit reconstructs FP64 fields and energy, then checks every population constraint and ordered electron hop. It retains the solver's legacy rounded constants. It checks agreement with the implemented model, not the physical constants' accuracy. Explicit checks remain active under `python -O`; rejected rows produce a nonzero exit status and recorded failures.

Rare hits have substantial uncertainty. The corpus and schedules reuse prior research, so this is integration qualification rather than an unseen-layout generalization study. Do not add percentages from independent experiments.

## Sources

Geometry credits and source hashes are embedded per fixture. Preserve [fiction's MIT notice](FICTION_LICENSE.txt). See [method attribution](../../docs/ATTRIBUTION.md) for QuickExact, ClusterComplete, PCG, and bounded-integer sampling.
