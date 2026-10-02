#!/usr/bin/env python3
"""Describe integrated TTS coverage without imputing zero-hit cells."""
import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
summary = json.loads((args.output / "summary.json").read_text())
fixtures = json.loads(Path(__file__).with_name("cases.json").read_text())
indexed = {(row["case"], row["arm"]): row for row in summary}
z = 1.959963984540054
for row in summary:
    for target in row["targets"].values():
        n, k = target["jobs"], target["hits"]
        p = k / n
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        target["wilson95_job_probability"] = [max(0, center - radius), min(1, center + radius)]

comparisons = []
for name, before, after in [
    ("portable_to_fixed_blas", "fixed_portable", "fixed_blas"),
    ("portable_to_combined", "fixed_portable", "combined"),
    ("fixed_to_shared_blas", "fixed_blas", "shared_blas"),
    ("legacy_to_combined", "legacy", "combined"),
    ("legacy_tuned_to_combined", "legacy_tuned", "combined"),
]:
    for corpus in ["large10", "original4", "fontes6", "exact3", "all13"]:
        cases = [c for c in fixtures if corpus == "all13" or
                 (corpus == "large10" and not c["certified_exact_target"]) or
                 (corpus == "original4" and c["name"].startswith("transfer_")) or
                 (corpus == "fontes6" and c["name"].startswith("continued_fontes")) or
                 (corpus == "exact3" and c["certified_exact_target"])]
        for target_name in ["initial", "midpoint", "strong"]:

            cells, undefined = [], []
            for c in cases:
                selected = "shared_blas" if c["cluster_size"] == 10 else "fixed_blas"
                keys = c["targets"]
                target_key = "frozen_strong_target"
                if not c["certified_exact_target"] and target_name == "initial":
                    target_key = "earlier_target" if "earlier_target" in keys else "initial_witness"
                elif not c["certified_exact_target"] and target_name == "midpoint":
                    target_key = "midpoint" if "midpoint" in keys else "arithmetic_midpoint"
                lhs = indexed[c["name"], before]["targets"][target_key]
                rhs = indexed[c["name"], selected if after == "combined" else after]["targets"][target_key]
                if lhs["tts997_ms"] is None or rhs["tts997_ms"] is None:
                    undefined.append(c["name"])
                else:
                    cells.append(dict(case=c["name"], frozen_target_key=target_key, before_ms=lhs["tts997_ms"],
                                      after_ms=rhs["tts997_ms"]))
            before_sum = sum(c["before_ms"] for c in cells)
            after_sum = sum(c["after_ms"] for c in cells)
            complete = not undefined
            comparisons.append(dict(
                comparison=name, corpus=corpus, target=target_name,
                finite_cases=len(cells), total_cases=len(cases), undefined_cases=undefined,
                all_case_sum_tts_before_ms=before_sum if complete else None,
                all_case_sum_tts_after_ms=after_sum if complete else None,
                all_case_reduction_percent=100 * (1 - after_sum / before_sum)
                    if complete and before_sum else None,
                descriptive_common_finite_cases=cells,
                common_finite_before_ms=before_sum, common_finite_after_ms=after_sum,
                common_finite_reduction_percent=100 * (1 - after_sum / before_sum)
                    if before_sum else None))

result = dict(
    cells=summary, comparisons=comparisons,
    combined_policy="Preselected shared BLAS for K10 cases and fixed BLAS for K6 cases.",
    target_aliases="Initial uses earlier_target/initial_witness; midpoint uses midpoint/arithmetic_midpoint; strong uses frozen_strong_target. Exact controls use their one certified target for all three aliases.",
    probability_interval="Two-sided Wilson 95% interval for observed complete-job success; not a TTS interval.",
    interpretation="Legacy comparators use integrated production lifecycle and corrected physical validation. Large targets are frozen witnesses, not ground-state certificates. Undefined cells prevent unconditional corpus totals. Common-finite subsets are explicitly descriptive. No additive performance gains.")
(args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"cells": len(summary), "comparisons": len(comparisons)}))
