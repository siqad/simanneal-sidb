# Initial scoped Auto qualification

This report records the initial mu=-0.32 eV qualification. The [mu-range extension](MU_RANGE_AUTO.md) describes the subsequent interval and backend qualification. The [larger and wider-physics extension](WIDE_AUTO.md) records the current Auto proposal. The measurements below retain their original scope.

The coupled short budget uses 256 cycles, 8/16/64 independent restarts, and hop factor 2. It retains uniform hopping and the existing optimized search. It does not select an exact solver.

## Ground-state qualification

Nibi measurements compared the merged `b00df0755834645d1a9b38b98c1aa11bfc1d983a` search defaults with the explicit short budget. Both arms used the same 360 certified layouts, 240 complete calls per layout, portable arithmetic, and 16 allocated CPUs. Physics was `mu=-0.32 eV`, relative permittivity 5.6, screening length 5 nm, and zero external and fixed-charge fields. Layouts had 2–35 sites. Each call ran the entire configured restart ensemble.

| Complete-process TTS at 99.7% confidence | Existing budget | Short budget | Improvement |
| --- | ---: | ---: | ---: |
| Sum over all 360 layouts | 42.383539 s | 3.102937 s | 13.66× / 92.68% lower |
| Geometric mean | 0.087165 s | 0.008287 s | 10.52× / 90.49% lower |

TTS combines measured call time with the probability that a call reaches the certified ground-state floor. Every layout had hits in both arms. These are estimates from finite samples, not guarantees for an individual invocation. Twelve tiny controls had point regressions of 0.06–2.44%. The hardest 32-site layout reached its floor in 201/240 existing-budget calls and 119/240 short-budget calls; its estimated TTS still fell by 86.11%.

The complete-process timer includes process launch, input parsing, native solver preparation, search, result preparation, JSON formatting, output collection, and teardown. Independent state audits and benchmark journal writes are outside the timer. These results do not isolate production XML serialization or predict gains for an already loaded Python process.

## Scope and fallback

The production guard uses the measured physical point and portable optimized search configuration. It requires the stock non-budget temperature, cooling, and freezing settings. It excludes optional history, refinement, masking, and the probability cache. Each Auto field resolves independently; numeric assignments remain literal. Performance evidence applies to the coupled budget, not every mixed override.

Native physics `mu=-0.25 eV` remains on the existing budget. In a separate 360-layout comparison, one 29-site layout fell from 76/120 ground-state hits to zero. That result rejects unconditional activation. The positive-excluded subgroup was promising, but selecting it after observing outcomes would require fresh qualification. Larger inputs and Accelerate/OpenBLAS also retain existing budgets. No new density classifier is qualified.

Existing XML numeric requests, restart sentinel `-1`, and worker sentinel `0` retain their meanings. New scalar Auto sentinels are observable before construction. Callers must read `effectiveParams()` for resolved values. The engine description uses integer sentinels for compatibility with SiQAD's editor; XML also accepts `auto`.

The prior private Auto prototype compared separate binaries and failed its 5% cost gate. It is not production-cost evidence. The production implementation resolves requests once during initialization and reuses the existing finite-field scan.

## Production integration cost

On this arm64 Mac, 1,000 fresh pairs compared production Auto with explicit short-budget values in the same Release binary. Ten frozen layouts covered 2, 9, 12, 20, 29, 32, and 35 sites, both charge classes, and sparse/dense 12-site cases. Each layout used 100 pairs, one worker, portable arithmetic, balanced randomized arm order, and matching 32-bit seeds. Every operational setting, exported configuration, energy, validity audit, and logical counter matched. Independent audits rejected zero records.

| Auto / explicit cost ratio | Point estimate | One-sided 95% bootstrap upper bound |
| --- | ---: | ---: |
| Complete process, all pairs | 0.97949 | 1.01855 |
| Native phases, all pairs | 0.99945 | 1.00628 |
| Complete process, positive excluded everywhere | — | 1.01483 |
| Complete process, positive possible | — | 1.04594 |

The aggregate and charge-class upper bounds pass the focused 1.05 gate. Three individual process upper bounds exceed 1.05, reaching 1.20086 for the smallest case. Every individual native upper bound is at most 1.02898. Retain that process-time uncertainty; this focused check does not repeat the earlier eight-group Nibi cost study or establish an intrinsic Auto speedup.

Portable and Accelerate macOS CTest suites pass, including backend fallback. SWIG 4.5.1 with Python 3.13 passes request, literal assignment, metadata, and parameter-lifetime tests. Linux portable/OpenBLAS/SWIG qualification runs in the PR workflow.

## Reproducibility

The retained research catalog identifies the frozen input, source, target, seed, and raw timing artifacts. Confirmation report SHA-256: `794af36432e0f666c8659fb48e01e9b0b9f6e677635920928b74f5d042e9c9ea`. Native fallback report SHA-256: `6f77060cca06532a467802b87c0c54636c7868cc9e8bb1ac8205b5a9c60c25ae`.

The public integration tests cover C++ requests, size boundaries, physical and search exclusions, XML requests and metadata, Python ownership, invalid inputs, and seeded Auto/explicit equivalence. Solver physics, validators, worker selection, and optional algorithms remain unchanged.
