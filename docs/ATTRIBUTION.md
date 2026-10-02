# Method and dependency attribution

The conservative charge bounds independently adapt **QuickExact**, by Jan Drewniok, Marcel Walter, and Robert Wille, *The Need for Speed: Efficient Exact Simulation of Silicon Dangling Bond Logic*, ASP-DAC 2024 ([manuscript](https://arxiv.org/abs/2308.04487)). The adaptation uses SimAnneal's charge signs, external and fixed fields, and numerical tolerances.

The local refinement uses ideas from **ClusterComplete**, by Willem Lambooy, Jan Drewniok, Marcel Walter, and Robert Wille, *Mastering the Exponential Complexity of Exact Physical Simulation of Silicon Dangling Bonds*, ASP-DAC 2026 ([manuscript](https://www.cda.cit.tum.de/files/eda/2026_aspdac_mastering_exact_simulation_of_silicon_dangling_bonds.pdf)). It enumerates small clusters while holding other charges fixed. Bounded repair, candidate selection, and the shared fixed/shifted strategy are heuristic adaptations. They do not provide globally exact simulation.

The reduced benchmark includes geometry from [fiction](https://github.com/cda-tum/fiction). Each fixture retains its source revision, path, transformations, and hash. Its MIT notice is in [FICTION_LICENSE.txt](../benchmarks/production_tts/FICTION_LICENSE.txt). Threshold provenance is separate from geometry provenance. Large-layout witness energies are not ground-state certificates.

The PCG XSH-RR engine adapts Melissa O'Neill's Apache-2.0 [pcg-c-basic](https://github.com/imneme/pcg-c-basic). Its copyright and license notice remain in `src/search_rng.h`. Integer sampling uses Daniel Lemire's multiplication and rejection method, *Fast Random Integer Generation in an Interval* ([manuscript](https://arxiv.org/abs/1805.10941)).

Optional matrix operations use [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS). OpenBLAS is a separately discovered dependency, not a bundled binary. Distributors must retain its license and the notices for their selected runtime dependencies. Apple builds can retain the existing Accelerate backend. Neither numerical backend is an original algorithm from this work.
