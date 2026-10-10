# Mu-range Auto qualification

This report records the 2–35-site nominal-physics qualification. The [larger and wider-physics extension](WIDE_AUTO.md) records subsequent qualification.

The range extension couples 256 cycles, 8/16/64 restarts, and hop factor 2.
It retains uniform hopping and the existing cooling/freezing schedule.
It also retains the best strict-valid state visited at cycle boundaries.
Explicit caller budgets and legacy search remain unchanged.

The Nibi comparison used base `a26ba61097275c6c0abb13ee252ecbe2333f3b4d`,
connector `766910907559e53b3e2b8704d04e5d387a5e521e`, GCC 12.3, and four CPUs per call.
Frozen splits contained 24 training geometries, 24 validation geometries,
one known diagnostic, and 311 reserved geometries.

The full comparison included 360 published random geometries with 2–35 sites,
eight mu values (-.32, -.30, -.28, -.26, -.25, -.24, -.22, -.20),
epsilon 5.6, screening length 5 nm, and zero external/fixed fields.
Each policy received 48 fresh calls per physical input, without target-aware stopping.
All 2,880 inputs had ground-state hits in all four policies.
The independent export audits found no invalid states or process failures.

| Scope | Native summed TTS gain | Native geometric gain | Process summed gain | Process geometric gain |
| --- | ---: | ---: | ---: | ---: |
| All 360 geometries × eight mu values | 51.00× | 37.27× | 21.93× | 15.24× |
| Reserved 311 geometries × eight mu values | 53.00× | 38.43× | 22.82× | 15.84× |

These gains compare current merged Auto against the retention fix plus short budgets.
At mu=-.32, current Auto already uses the short budget: native summed gain was 1.012×,
and geometric gain was 0.991×. Most improvements avoid the 10,000-cycle fallback elsewhere.
Seven inputs regressed by more than 20% in native TTS; all had 48/48 hits.
Five penalties were 13–18 microseconds; two were approximately 1.4–1.5 milliseconds.
Five short-policy inputs had only 2–10 hits, so their individual TTS estimates remain uncertain.

TTS99.7 uses mean call time times max(1, log(.003)/log1p(-hits/calls)).
Zero-hit TTS remains undefined. No full input needed a zero-hit exclusion here.
Native time includes preparation, search, result retrieval, and solver destruction.
Process time also includes startup, input parsing, output formatting, and process teardown.
Independent audits and journal writes are excluded from both timers.

Targets used official full-coupled base-3 ClusterComplete from mnt.pyfiction 0.8.0,
with pair-coupling parity and independent state validation.
One wrapper rejected a population margin of -0.292 micro-eV under its 0.001 micro-eV gate.
An independent exhaustive 524,288-assignment search certified the same floor under
SimAnneal's unchanged 1 micro-eV validator. Its failed receipt and separate recovery remain recorded.

The separate Mac check used 192 validation inputs and 12,288 calls across short/warm
and portable/Accelerate backends. All inputs retained ground-state coverage, with no invalid exports.
Accelerate short native summed TTS was 2.6% higher than portable on that subset.
This qualifies backend compatibility; it is not a whole-corpus Mac speedup measurement.

Warm and one-eighth-warm portfolios remain research comparisons.
Neither improved aggregate TTS over short on the reserved set.
No new density classifier, charge-class classifier, or user toggle is introduced.
This report does not cover more than 35 sites, nonzero fields, or other epsilon/screening values.
