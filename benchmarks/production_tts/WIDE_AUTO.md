# Larger and wider-physics Auto qualification

Auto retains uniform hopping and the stock temperature, cooling, and freezing settings.
It does not select an exact solver or remove transient positive charge states.
The optimized profile also retains the best strict-valid state visited at cycle boundaries.
The tentative repair path and actual repair flags remain unchanged.

| Sites | Cycles per restart | Independent restarts | Hop factor |
| --- | ---: | ---: | ---: |
| 2–9 | 256 | 8 | 2 |
| 10–25 | 256 | 16 | 2 |
| 26–35 | 256 | 64 | 2 |
| 36–62 | 512 | 128 | 2 |

The common guard requires mu in [-0.32,-0.20] eV and zero external/fixed fields.
It retains the existing search-configuration guards described in [optimized search](../../docs/optimized_search.md#validity-and-budgets).
Portable, Auto, and available Accelerate backends qualify; explicit OpenBLAS does not.
Explicit caller budgets remain literal. Worker limits follow hardware and CPU affinity, independently of restart counts.

At epsilon 5.6 and screening 5 nm, both preparation charge classes qualify.
Otherwise, epsilon and screening in [1,10] qualify only when every prepared site excludes positive charge.
Screening uses nanometres. The proof uses the extreme all-other-sites-negative bound and existing model tolerances.
It never classifies a layout from observed charges. Outside these conditions, Auto retains conservative budgets.

## Larger nominal-physics inputs

The separate larger-layout study used 22 canonical cases with 36–69 sites and eight mu values.
These comprised 18 source SQD layouts and four mapped MNTBench cases, not recovered ClusterComplete paper instances.
Full-coupled ClusterComplete certified 161 physical inputs; 15 parameter-specific oracle calls timed out after 600 seconds.
All inputs used epsilon 5.6, screening 5 nm, zero fields, and unchanged model tolerances.

The initial 256-cycle/64-restart pilot missed a target that the baseline reached.
A limited budget comparison selected 512 cycles and 128 restarts before fresh full measurement.
Each certified input received 48 paired calls per policy with four workers and randomized policy order.
All exported states passed independent energy, population, and hopping audits.

Within 36–62 sites, baseline reached 136/152 certified inputs; the selected budget reached 151/152.
On exactly 135 common-finite inputs, native summed/geometric TTS improved 59.82×/32.68×.
Complete-process summed/geometric TTS improved 46.36×/25.30× on the same set.
The sole new zero-hit case had five candidate hits in a separately frozen 128-call confirmation.
Those confirmation calls remain separate from the headline estimates.
The 65–69-site evidence covered only three mu values, so Auto stops at 62 sites.

## Wider-physics production comparison

The frozen baseline was `a26ba61097275c6c0abb13ee252ecbe2333f3b4d`.
The pinned connector was `766910907559e53b3e2b8704d04e5d387a5e521e`.
The candidate combined strict-valid retention with actual production Auto selection.
This study did not rerun the earlier 360-layout or 22-case studies.

The panel contained 318 physical inputs from 28 geometry groups.
It included official Bestagon input patterns, published random layouts, and eight canonical 36–62-site controls.
The main grid fixed mu=-0.32 eV, varying epsilon and screening over structured points in [1,10].
Additional samples covered the positive-exclusion boundary and 16 corresponding mu=-0.20 points.
This is sampled qualification of an interval guard, not an exhaustive continuous operational-domain map.

ClusterComplete supplied 306 exact targets, including 28 compatible cached certificates.
Twelve positive-possible inputs retained recorded 600-second oracle timeouts and were excluded.
Nibi ran 48 fresh paired calls per policy and certified input: 29,376 calls with four matched workers.
There were no process failures, invalid exports, or exact-floor contradictions.

| Scope | Certified / common-finite | Native sum / geometric speedup | Process sum / geometric speedup |
| --- | ---: | ---: | ---: |
| All certified inputs | 306 / 295 | 2.28× / 5.61× | 2.26× / 3.48× |
| Positive excluded everywhere | 199 / 198 | 252.16× / 12.53× | 142.83× / 6.17× |
| Positive possible | 107 / 97 | 1.10× / 1.09× | 1.09× / 1.08× |
| 36–62 sites | 93 / 84 | 2.52× / 8.01× | 2.51× / 7.05× |

Baseline reached 295/306 targets; the candidate reached 298/306.
No baseline-hit input became zero-hit. All 199 positive-excluded inputs had candidate hits.
Eight positive-possible inputs remained zero-hit under both policies.
Aggregates exclude those eight inputs and the three candidate-only inputs.
They are finite-set gains, not whole-panel gains.
The large positive-excluded summed multiplier is dominated by difficult cases with rare baseline hits.
Individual low-hit TTS estimates remain uncertain.

All 5,040 paired calls outside the new Auto guard preserved reported search counters.
Their candidate strict-valid minimum never exceeded the corresponding baseline minimum.
The pilot sampled all geometry groups; full fresh trials were not untouched-geometry holdouts.
Do not multiply gains from the separate studies.

TTS99.7 equals mean call time times max(1, ln(0.003)/ln(1−hits/calls)).
All-hit samples use one call; zero-hit estimates remain undefined.
Native timing includes preparation, search, results, and destruction.
Process timing additionally includes startup, parsing, serialization, and exit.
Independent audits and journal writes remain outside both timers. No target-aware early stopping was used.

## Interfaces, evidence, and attribution

Mac Portable and Accelerate builds passed their CTest, CLI, and Python/SWIG profile and batch checks.
A separate nominal-physics Mac spot check audited 536 calls on eight 36–62-site inputs.
Each backend had 70/128 baseline hits and 97/128 candidate hits.
One 53-site input lost per-call hits but improved complete-process TTS 10.89×.
Auto/explicit results matched at 1, 4, and 8 workers on both backends.
This is spot evidence, not a full wider-physics Mac performance measurement.

Raw attempts, certificates, source identities, common-finite IDs, and uncertainty tables remain in the research catalog.
The full archive SHA-256 is `2c385975289fa805f64b25a8f97b2fe28dbcc694e15e90f1ef153551bfa6781a`.
Its exact member set and all 1,586 member checksums passed verification.
The production `src/simanneal.cc` SHA-256 is `c653661543259c72e62c7fd0a85877f5983a5a940de60d4fdb8bd7b0aa73c684`.

Credit [Munich Nanotech Toolkit fiction](https://github.com/cda-tum/fiction), mnt.pyfiction, and the QuickExact/ClusterComplete authors.
Jan Drewniok's QuickExact charge-exclusion bound supplies the method adapted by Auto.
ClusterComplete supplies parameter-specific exact reference certificates.
Bestagon source provenance uses fiction revision `3698e59825804b2aa57d639b041a4acdc371f158`.
See [method attribution](../../docs/ATTRIBUTION.md) and the retained [fiction license](FICTION_LICENSE.txt).

SimAnneal remains heuristic. Ground-state TTS does not qualify exhaustive enumeration of degenerate ground states.
Nonzero fields, sizes above 62, and other physics remain outside this Auto extension.
