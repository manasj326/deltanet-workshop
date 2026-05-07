"""
Head Ablation Sweep

Loads the model once and ablates every (layer, head) pair, generating a
response for each. Records which heads produce a wrong answer.

Sweeps both DeltaNet (linear_attn) and standard self-attention layers.
By default covers all layers; restrict with --layers / --linear-only.

Supports multiple prompts: pass --prompts and --expected as paired lists of
equal length. The script writes per-prompt CSVs + heatmaps and (if multiple
prompts) a combined CSV/heatmap showing how many prompts each (layer, head)
breaks.

Usage:
  # Single prompt — sweep ALL heads (linear + self-attn)
  python head_sweep.py --prompts prompts/recall_easy.txt --expected 9

  # Linear-attn layers only (old default)
  python head_sweep.py --prompts prompts/recall_easy.txt --expected 9 --linear-only

  # Multi-prompt
  python head_sweep.py \
    --prompts prompts/recall_easy.txt prompts/recall_2dup.txt prompts/recall_3dup.txt \
    --expected 9 9 9
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from common import setup_model, tokenize_prompt, prompt_name
from head_ablation import ablate_head, layer_attn_info


def generate(model, tokenizer, input_ids, attention_mask, max_new_tokens):
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    new_tokens = outputs[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def is_correct(response, expected):
    norm = response.replace(",", " ").replace(".", " ").replace("?", " ").replace("!", " ")
    tokens = norm.split()
    return expected in tokens


def build_layer_plan(model, layers, head_filter, linear_only):
    """
    For each requested layer, decide which heads to sweep based on its
    attention type. Returns a list of (layer_idx, attn_type, [head_indices]).
    """
    plan = []
    for l in layers:
        layer_module = model.model.layers[l]
        attn_type, n = layer_attn_info(layer_module)
        if attn_type is None:
            continue
        if linear_only and attn_type != "linear":
            continue
        heads = head_filter if head_filter is not None else list(range(n))
        # Clamp to actual head count for this layer
        heads = [h for h in heads if h < n]
        if heads:
            plan.append((l, attn_type, heads))
    return plan


def sweep_one_prompt(model, tokenizer, prompt_file, expected, layer_plan,
                     max_tokens, device):
    pname = prompt_name(prompt_file)
    input_ids, attention_mask = tokenize_prompt(prompt_file, tokenizer, device)

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt_file}  (expected={expected!r})")
    print(f"{'='*60}")
    baseline = generate(model, tokenizer, input_ids, attention_mask, max_tokens)
    baseline_ok = is_correct(baseline, expected)
    print(f"Baseline: {baseline!r}  (correct={baseline_ok})")

    output_dir = os.path.join("plots", pname)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "head_sweep.csv")

    results = []
    with open(csv_path, "w") as f:
        f.write("layer,attn_type,head,is_correct,response\n")
        for layer_idx, attn_type, heads in layer_plan:
            for h in heads:
                cleanup = ablate_head(model, layer_idx, h)
                try:
                    resp = generate(model, tokenizer, input_ids, attention_mask, max_tokens)
                finally:
                    cleanup()
                ok = is_correct(resp, expected)
                results.append((layer_idx, attn_type, h, resp, ok))
                resp_csv = resp.replace(",", "\\,").replace("\n", "\\n")
                f.write(f"{layer_idx},{attn_type},{h},{int(ok)},{resp_csv}\n")
                f.flush()
                marker = "✓" if ok else "✗"
                tag = "LA" if attn_type == "linear" else "SA"
                print(f"  L{layer_idx:2d}[{tag}] H{h:2d}  [{marker}]  {resp!r}")

    print(f"  -> saved to {csv_path}")
    return baseline, baseline_ok, results


def _sort_row_indices(row_values):
    """
    Return the head indices for one row, sorted so existing heads come first
    (descending by value), then NaN-filled slots. NaNs go to the right.
    """
    n = len(row_values)
    valid = [i for i in range(n) if not np.isnan(row_values[i])]
    invalid = [i for i in range(n) if np.isnan(row_values[i])]
    valid.sort(key=lambda i: -row_values[i])
    return valid + invalid


def plot_correctness_heatmap(layer_plan, results, output_path, title,
                             baseline_ok=None):
    """
    Per-row sorted heatmap. Each row is one layer; cells are sorted by
    correctness (wrong first, since wrong is the interesting signal),
    with the actual head index shown inside each cell. Gray = head
    doesn't exist for that layer.
    """
    layers = [l for l, _, _ in layer_plan]
    layer_types = {l: t for l, t, _ in layer_plan}
    max_heads = max(max(h) for _, _, h in layer_plan) + 1

    grid = np.full((len(layers), max_heads), np.nan)
    lookup = {(l, h): ok for l, _, h, _, ok in results}
    for i, l in enumerate(layers):
        valid = layer_plan[i][2]
        for h in valid:
            if (l, h) in lookup:
                grid[i, h] = 1.0 if lookup[(l, h)] else 0.0

    # Sort each row: wrong (0) first, then correct (1), then NaN
    sorted_grid = np.full_like(grid, np.nan)
    sorted_heads = np.full(grid.shape, -1, dtype=int)
    for i in range(len(layers)):
        order = _sort_row_indices(-grid[i])  # negate so 0 (wrong) sorts highest
        for col, head_idx in enumerate(order):
            sorted_grid[i, col] = grid[i, head_idx]
            sorted_heads[i, col] = head_idx

    fig, ax = plt.subplots(figsize=(max(8, max_heads * 0.55),
                                    max(4, len(layers) * 0.32)))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("lightgray")
    masked = np.ma.masked_invalid(sorted_grid)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(max_heads))
    ax.set_xticklabels([str(c) for c in range(max_heads)])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(
        [f"L{l} ({'LA' if layer_types[l] == 'linear' else 'SA'})" for l in layers]
    )
    ax.set_xlabel("Rank within row (most-broken first)")
    ax.set_ylabel("Layer")

    full_title = title
    if baseline_ok is not None:
        full_title += f"  (baseline {'✓' if baseline_ok else '✗'})"
    ax.set_title(full_title)

    for i in range(len(layers)):
        for col in range(max_heads):
            v = sorted_grid[i, col]
            if not np.isnan(v):
                head_idx = sorted_heads[i, col]
                ax.text(col, i, f"H{head_idx}", ha="center", va="center",
                        fontsize=7, color="white")

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["wrong", "correct"])
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  -> heatmap saved to {output_path}")


def plot_failure_heatmap(layer_plan, break_map, n_prompts, output_path, title):
    """
    Per-row sorted heatmap of fail rate (0..1). Each row is one layer; cells
    are sorted by fail rate descending (most-broken head leftmost). Cell text
    shows the head index so you can see which head sits at each rank.
    """
    layers = [l for l, _, _ in layer_plan]
    layer_types = {l: t for l, t, _ in layer_plan}
    max_heads = max(max(h) for _, _, h in layer_plan) + 1

    grid = np.full((len(layers), max_heads), np.nan)
    for i, l in enumerate(layers):
        valid = layer_plan[i][2]
        for h in valid:
            grid[i, h] = len(break_map.get((l, h), [])) / n_prompts

    # Sort each row by fail rate descending (NaNs to the right)
    sorted_grid = np.full_like(grid, np.nan)
    sorted_heads = np.full(grid.shape, -1, dtype=int)
    for i in range(len(layers)):
        order = _sort_row_indices(grid[i])
        for col, head_idx in enumerate(order):
            sorted_grid[i, col] = grid[i, head_idx]
            sorted_heads[i, col] = head_idx

    fig, ax = plt.subplots(figsize=(max(8, max_heads * 0.55),
                                    max(4, len(layers) * 0.32)))
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("lightgray")
    masked = np.ma.masked_invalid(sorted_grid)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(max_heads))
    ax.set_xticklabels([str(c) for c in range(max_heads)])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(
        [f"L{l} ({'LA' if layer_types[l] == 'linear' else 'SA'})" for l in layers]
    )
    ax.set_xlabel("Rank within row (most-broken first)")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    for i in range(len(layers)):
        for col in range(max_heads):
            v = sorted_grid[i, col]
            if not np.isnan(v):
                head_idx = sorted_heads[i, col]
                count = int(round(v * n_prompts))
                ax.text(col, i, f"H{head_idx}\n{count}",
                        ha="center", va="center", fontsize=6,
                        color=("white" if v > 0.5 else "black"))

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"fail rate (out of {n_prompts})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  -> heatmap saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablate every head and check the response")
    parser.add_argument("--prompts", type=str, nargs="+", required=True,
                        help="One or more prompt files")
    parser.add_argument("--expected", type=str, nargs="+", required=True,
                        help="Expected answer for each prompt (must match --prompts in length)")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer indices to sweep (default: all layers with attention)")
    parser.add_argument("--heads", type=int, nargs="+", default=None,
                        help="Head indices to sweep (default: all per layer; clamped to layer's head count)")
    parser.add_argument("--linear-only", action="store_true",
                        help="Only sweep DeltaNet (linear_attn) layers; skip standard self-attn")
    parser.add_argument("--combined-tag", type=str, default=None,
                        help="Tag for combined output dir (default: derived from prompt names)")
    args = parser.parse_args()

    if len(args.prompts) != len(args.expected):
        raise ValueError(
            f"--prompts ({len(args.prompts)}) and --expected ({len(args.expected)}) "
            "must have the same length"
        )

    model, tokenizer, device = setup_model()
    layers = args.layers if args.layers else list(range(len(model.model.layers)))

    layer_plan = build_layer_plan(model, layers, args.heads, args.linear_only)
    if not layer_plan:
        raise RuntimeError("No layers to sweep — check --layers / --linear-only")

    total_ablations = sum(len(h) for _, _, h in layer_plan) * len(args.prompts)
    print(f"Sweeping {len(layer_plan)} layers x {len(args.prompts)} prompts = "
          f"{total_ablations} ablations "
          f"(linear={sum(1 for _, t, _ in layer_plan if t == 'linear')}, "
          f"self={sum(1 for _, t, _ in layer_plan if t == 'self')})")

    break_map = {}
    per_prompt_summary = []

    for prompt_file, expected in zip(args.prompts, args.expected):
        baseline, baseline_ok, results = sweep_one_prompt(
            model, tokenizer, prompt_file, expected,
            layer_plan, args.max_tokens, device,
        )
        pname = prompt_name(prompt_file)
        n_failures = sum(1 for _, _, _, _, ok in results if not ok)
        per_prompt_summary.append((pname, baseline, baseline_ok, n_failures, len(results)))
        for layer_idx, _, h, _, ok in results:
            if not ok:
                break_map.setdefault((layer_idx, h), []).append(pname)

        # Per-prompt heatmap
        heatmap_path = os.path.join("plots", pname, "head_sweep_heatmap.png")
        plot_correctness_heatmap(
            layer_plan, results, heatmap_path,
            title=f"Head ablation correctness — {pname}",
            baseline_ok=baseline_ok,
        )

    # Combined report (only meaningful with multiple prompts)
    if len(args.prompts) > 1:
        tag = args.combined_tag or "+".join(prompt_name(p) for p in args.prompts)
        combined_dir = os.path.join("plots", "head_sweep_combined", tag)
        os.makedirs(combined_dir, exist_ok=True)
        combined_csv = os.path.join(combined_dir, "head_sweep_combined.csv")

        all_pairs = [(l, h) for l, _, heads in layer_plan for h in heads]

        with open(combined_csv, "w") as f:
            f.write("layer,attn_type,head,n_failed,n_prompts,fail_rate,prompts_failed\n")
            type_lookup = {l: t for l, t, _ in layer_plan}
            for layer_idx, h in all_pairs:
                failed_on = break_map.get((layer_idx, h), [])
                n_failed = len(failed_on)
                fail_rate = n_failed / len(args.prompts)
                f.write(f"{layer_idx},{type_lookup[layer_idx]},{h},{n_failed},"
                        f"{len(args.prompts)},{fail_rate:.3f},"
                        f"{';'.join(failed_on)}\n")

        plot_failure_heatmap(
            layer_plan, break_map, len(args.prompts),
            os.path.join(combined_dir, "head_sweep_heatmap.png"),
            title=f"Heads broken across {len(args.prompts)} prompts ({tag})",
        )

        print(f"\n{'='*60}")
        print(f"COMBINED SUMMARY across {len(args.prompts)} prompts")
        print(f"{'='*60}")
        print(f"\nPer-prompt baseline + failure counts:")
        for pname, baseline, ok, nfail, ntotal in per_prompt_summary:
            mark = "✓" if ok else "✗"
            print(f"  {pname:25s}  baseline={baseline!r:30s} [{mark}]  "
                  f"failures={nfail}/{ntotal}")

        common = sorted(break_map.items(), key=lambda kv: -len(kv[1]))
        print(f"\nHeads ranked by # prompts they broke (out of {len(args.prompts)}):")
        for (layer_idx, h), prompts_failed in common:
            print(f"  L{layer_idx:2d} H{h:2d}  ({len(prompts_failed)}/{len(args.prompts)})"
                  f"  prompts: {', '.join(prompts_failed)}")
        print(f"\nCombined CSV saved to {combined_csv}")
