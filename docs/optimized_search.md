# Validated repair and bounded local refinement

SimAnneal remains a heuristic. A physically valid result is not a certificate of global optimality. The optimized profile is the default. It enables PCG32 sampling, bounded repair, and the validated singleton shortcut. Select `search_profile=legacy` to restore legacy profile defaults. Refinement and transient domain masking remain opt-in.

## Search settings

Set these parameters in the input XML `<sim_params>` section. SiQAD exposes the same fields through `simanneal.physeng`.

| Parameter | Default | Effect |
|---|---|---|
| `search_profile` | `optimized` | `optimized` enables PCG32 sampling, bounded repair, and the singleton shortcut. |
| `random_backend` | `auto` | Resolve from the profile, or select `mt` or `pcg32` explicitly. |
| `repair` | `profile` | Override the profile with `true` or `false`. |
| `singleton_shortcut` | `profile` | Validate the sole conservatively admissible configuration before scheduling restarts. |
| `probability_shortcuts` | `true` | Avoid exponential evaluation when the finite random grid already determines the decision. Preserve random draws and the zero endpoint. |
| `population_probability_cache` | `false` | For at least 64 sites, cache inverse temperature and the shortcut bound per population update. Rounding can change search paths. Preserve draws, schedules, and hopping acceptance. |
| `transient_domain_mask` | `false` | Apply final-state charge exclusions during annealing. This changes search paths and can help or harm convergence. |
| `refinement` | `none` | Select `k6`, `k10`, or `shared`. The latter uses fixed and shifted ten-site clusters. |
| `refinement_candidates` | `8` | Refine at most this many distinct, lowest-energy valid candidates. |
| `refinement_rounds` | `1` | Bound sequential rounds per candidate. More rounds do not guarantee lower time to solution. |
| `refinement_trials` | `1` | Bound the number of lower-energy cluster proposals repaired per round. |
| `population_backend` | `auto` | Existing portable/Accelerate selection, or explicit `openblas`/`openblas_symmetric`. |

For an initial refinement experiment, use `refinement=k6`. Compare K10 and shared refinement at the same target and complete-job timing boundary. Tune schedule lengths separately if needed. No profile is selected automatically from a filename or site count.

The shared strategy refines identical annealing candidates independently with fixed and shifted cluster geometries. The shift uses a portable, unbiased seed mapping and places shifted centers between fixed centers. Layouts with fewer than 64 sites intentionally use a single-cache fallback. Metadata reports the actual nonempty geometry count and the fallback flag. It retains the best valid result. Some layouts gain success probability; others only incur extra work. Keep the fixed strategy available.

The domain bounds include external potentials and fixed charges. Final-state exclusions do not prove that removing transient states preserves useful annealing paths. The transient mask remains a separate option.

## Validity and budgets

Each accepted repair/refinement result passes the common full population and ordered-hop validator. Neutral-to-positive hopping uses the same electron-transfer direction as the Hamiltonian. Results must have finite energy.

Repair permits at most eight population passes and four times the site count in downhill hops. A capped stage retains a validated incumbent. Budget exhaustion does not mean that the incumbent is invalid or globally optimal.

Geometry caches belong to the solver job. The persistent cluster-cache limit is 8 MiB per geometry and 16 MiB for the shared pair. It counts reserved cluster headers and pattern payload. Geometry/engine owner objects and temporary scratch are outside that accounting. Deduplication has a separate 8 MiB limit. These bounds do not cap allocator overhead, model matrices, or parallel scratch storage.

One live `SimAnneal` object owns the active model. Keep that object alive while reading or exporting its results. A second live object is rejected. Independent concurrent jobs still require separate processes. Internal workers share one immutable model. The static compatibility API must not be mutated during a job.

XML metadata reports the effective profile, RNG, numerical backend, executed restarts, singleton use, and repair/refinement budget status. A singleton result executes zero restarts. Ordinary restart records remain intact. A strictly better refinement result is an additional exported record. Occurrence counts count exported records, including history when enabled; they are not target-hit probabilities.

Tidy export deduplicates initialized records before full validation. It retains the first record metadata and original order, then recalculates valid energies.
Export can validate candidates in parallel when there are at least 32 unique records, 128 sites, and four million estimated pair operations.
Workers never exceed the configured worker count or one per eight unique records, rounded up. Smaller exports remain serial.
Keep the model and result records unchanged until export returns. Async launch failure completes the unassigned records serially.

For layouts with at least 64 sites, the optional population probability cache uses multiplication by a cached reciprocal for finite positive normal temperatures.
Smaller layouts retain the original sampler even when the option is enabled.
Subnormal, zero, negative, or nonfinite temperatures retain division. This option changes floating-point rounding and can change trajectories in either profile.
The default remains disabled. Python exposes the same `SimParams.population_probability_cache` field, and XML metadata reports the effective value.

## Numerical backends

Portable arithmetic and Apple Accelerate remain available. To compile the optional OpenBLAS backend:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSIMANNEAL_ENABLE_OPENBLAS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Install a supported OpenBLAS development package first. For a custom installation, set `SIMANNEAL_OPENBLAS_ROOT`, or supply `SIMANNEAL_OPENBLAS_INCLUDE_DIR` and `SIMANNEAL_OPENBLAS_LIBRARY` explicitly. CMake checks CBLAS operations and thread-control symbols. Do not use private NumPy wheel symbols.

OpenBLAS uses one numerical thread per annealing worker during a search. SimAnneal restores the previous OpenBLAS thread count when the search returns or throws. Construction does not change that setting. Thread control remains library-global while the search runs; applications must coordinate unrelated concurrent OpenBLAS use. Changing the numerical backend can change floating-point reduction order and search trajectories. The symmetric backend requires the solver's symmetric interaction matrix. Apple builds cannot enable Accelerate and OpenBLAS together because they export identical CBLAS symbols. For OpenBLAS on Apple, set `SIMANNEAL_ENABLE_ACCELERATE=OFF`; CMake rejects a build with both enabled.

Fast-math is disabled. `SIMANNEAL_NATIVE_ARCH=ON` enables host-specific instructions with supported compilers. Such binaries are not portable distribution artifacts.

## Measurement and credit

Use the [production benchmark](../benchmarks/production_tts/README.md) for complete-job TTS estimates. Do not add component percentages from separate experiments. Preserve zero-hit cells and distinguish exact targets from validated witnesses.

See [method attribution](ATTRIBUTION.md) for QuickExact, ClusterComplete, fiction, PCG, and Lemire credit.
