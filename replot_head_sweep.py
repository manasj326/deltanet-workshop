"""
Regenerate head-sweep heatmaps from existing CSVs (no re-running the sweep).

Usage:
  # Per-prompt heatmap
  python replot_head_sweep.py --csv plots/recall_1/head_sweep.csv

  # Combined heatmap (multi-prompt)
  python replot_head_sweep.py --combined-csv plots/head_sweep_combined/head_sweep/head_sweep_combined.csv

  # Several at once
  python replot_head_sweep.py \
    --csv plots/recall_1/head_sweep.csv plots/recall_2/head_sweep.csv \
    --combined-csv plots/head_sweep_combined/head_sweep/head_sweep_combined.csv
"""

import os
import csv
import argparse
from collections import defaultdict

from head_sweep import plot_correctness_heatmap, plot_failure_heatmap


def load_per_prompt_csv(path):
    """Returns (layer_plan, results) compatible with plot_correctness_heatmap."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((
                int(r["layer"]),
                r["attn_type"],
                int(r["head"]),
                r["response"],
                bool(int(r["is_correct"])),
            ))
    heads_by_layer = defaultdict(list)
    type_by_layer = {}
    for layer, attn_type, head, _, _ in rows:
        heads_by_layer[layer].append(head)
        type_by_layer[layer] = attn_type
    layer_plan = [
        (l, type_by_layer[l], sorted(set(heads_by_layer[l])))
        for l in sorted(heads_by_layer)
    ]
    return layer_plan, rows


def load_combined_csv(path):
    """Returns (layer_plan, break_map, n_prompts) for plot_failure_heatmap."""
    heads_by_layer = defaultdict(list)
    type_by_layer = {}
    break_map = {}
    n_prompts = None
    with open(path) as f:
        for r in csv.DictReader(f):
            layer = int(r["layer"])
            head = int(r["head"])
            heads_by_layer[layer].append(head)
            type_by_layer[layer] = r["attn_type"]
            if n_prompts is None:
                n_prompts = int(r["n_prompts"])
            failed = [s for s in r["prompts_failed"].split(";") if s]
            if failed:
                break_map[(layer, head)] = failed
    layer_plan = [
        (l, type_by_layer[l], sorted(set(heads_by_layer[l])))
        for l in sorted(heads_by_layer)
    ]
    return layer_plan, break_map, n_prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, nargs="+", default=[],
                        help="Per-prompt head_sweep.csv file(s)")
    parser.add_argument("--combined-csv", type=str, nargs="+", default=[],
                        help="Combined head_sweep_combined.csv file(s)")
    args = parser.parse_args()

    if not args.csv and not args.combined_csv:
        parser.error("provide at least one --csv or --combined-csv")

    for path in args.csv:
        layer_plan, results = load_per_prompt_csv(path)
        out = os.path.join(os.path.dirname(path), "head_sweep_heatmap.png")
        prompt_label = os.path.basename(os.path.dirname(path))
        plot_correctness_heatmap(
            layer_plan, results, out,
            title=f"Head ablation correctness — {prompt_label}",
        )

    for path in args.combined_csv:
        layer_plan, break_map, n_prompts = load_combined_csv(path)
        out = os.path.join(os.path.dirname(path), "head_sweep_heatmap.png")
        tag = os.path.basename(os.path.dirname(path))
        plot_failure_heatmap(
            layer_plan, break_map, n_prompts, out,
            title=f"Heads broken across {n_prompts} prompts ({tag})",
        )
