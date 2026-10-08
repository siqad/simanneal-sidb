# Default runtime optimizations (2026-10-08)

Four implementation changes preserve the search and become internal defaults:

1. Omit negative targets in the charge-class and bounded repair selectors. Every legal target has greater charge than its donor.
2. Transfer temporary parameter matrix storage explicitly. Boost uBLAS can copy ordinary matrix moves, so the native rvalue constructor swaps matrix storage. CLI reference forwarding avoids another matrix copy. Lvalue and Python callers retain reusable parameters.
3. Build symmetric geometry in parallel for at least512 sites. Concurrency is at most eight, the available CPUs and resolved worker limit. Small layouts, one-worker jobs and debug logging stay serial.
4. Schedule tidy export by actual unique-configuration matrix work. Remove the redundant32-configuration gate while retaining the existing4,000,000-work threshold, minimum128 sites, eight configurations per worker, strict validation, order and deduplication.

These changes do not introduce automatic exact-solver selection, transient masking, new refinement policies, different annealing budgets, or new user toggles.

## Combined measurements

| Platform and workload | Summed native runtime reduction |
|---|---:|
| Nibi research stack, all 123 layouts | 20.50% (95% interval 18.54–22.40%) |
| Nibi research stack, largest 4 layouts, 651–1211 sites | 30.80% (27.49–33.78%) |
| macOS production implementation, all 123 layouts | 13.23% (10.94–16.05%) |
| macOS production implementation, largest 4 layouts | 18.42% (16.39–20.22%) |
| macOS production implementation,119 layouts below 512 sites | 7.43% (3.60–12.59%) |

Nibi used Xeon 6972P, 16 allocated CPUs, single-thread OpenBLAS, 128 cycles, 16 restarts and 16 workers. Geometry used eight workers. The research comparison used 16 fresh paired jobs per layout and unchanged targets. Exact 106 controls were approximately neutral (0.96% slower). The global geometric-mean runtime reduction was only 1.00%; large layouts dominate summed runtime savings.

macOS used arm64, AppleClang 21, Boost 1.92, Release with strict floating-point arithmetic, portable population arithmetic, 128 cycles, 16 restarts and eight workers. Baseline: master 7feafb3996f9b66da5d46e5c780f1868f0b59d3d. Each layout had 64 fresh paired jobs, alternating baseline/default arm order and deterministically shuffling layout order. Seeds were 4200600000+pair*256 for pair 0..63. The original 8-pair pilot was retained separately and excluded from these results because short jobs showed timing noise. No parameters were tuned from these outcomes.

Timing includes native setup, invocation, strict result preparation and cleanup. It excludes process launch, input/output transfer and file serialization. Do not add component percentages or combine platform percentages. [Per-layout macOS measurements](runtime_defaults_mac.csv) include regressions as well as improvements; these are qualification timings, not a guarantee for every workload.

All 5904 Nibi outputs and counters matched across paired arms. All 15744 macOS outputs matched within 7872 paired comparisons, including configuration, energy, validity, effective settings and logical counters. Charge-space classification uses pre-simulation bounds, never observed output charges: 117 layouts exclude positive charges everywhere and six permit positive charges.

## TTS boundary

These are runtime savings. They do not establish a new finite whole-corpus ground-state TTS percentage. All 17 historical layouts had zero hits at the updated lowest best-known target in every final Nibi arm, including baseline. Those shared zeros are not implementation regressions. Exact controls and best-known witness targets remain separate. No targets were relaxed for this qualification.

## Production checks

- macOS Release native, affinity and XML/CLI CTest suites pass.
- Matrix storage identity, copying callers, overlap rejection, failed initialization recovery and actual interface forwarding are tested.
- SWIG 4.5/Python 3.12 runtime tests pass for reusable parameters, source-parameter deletion, sequential models, profiles and strict exported energies. Python does not expose the consuming overload.
- macOS Debug AddressSanitizer and UndefinedBehaviorSanitizer pass all CTest suites.
- Geometry tests compare matrix bits, fields, local selection and nonfinite detection around the 512-site threshold. Linux affinity tests cover large sparse masks, denied queries and explicit limits.
- Exact move tests compare both repair selectors against the original nested scan across ternary states, ties and nonfinite values.
- Linux portable/OpenBLAS/LTO and SWIG qualification runs in the PR CMake workflow.

The native interface overloads preserve ordinary source calls but change binary signatures; rebuild native clients with the updated library. The change does not modify SiQAD's submodule revision.

Full raw inputs, output rows, source snapshots and checksums remain in the gitignored primary-checkout research archive under `build-research/simanneal-optimization/2026-10-06/cpu-density-budget/four_hour_20261008`, with production qualification in `default-production-qualification/`. Physical input catalogue SHA256: `b0a85ea83519534837fa56b7ccceb6f62af7ef5c15bb20fdc7ee06da7eb8970c`.

Borrowed charge bounds, refinement and fixture methods retain [QuickExact, ClusterComplete and fiction attribution](../../docs/ATTRIBUTION.md). This PR does not add their unqualified pair-pruning prototype.
