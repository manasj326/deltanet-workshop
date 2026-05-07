"""
Duplicate Gate Analysis

Examines alpha (retention) and beta (update) gate values at the token positions
where the target word (e.g. "apple") is assigned a score, in prompts with
conflicting duplicate assignments. Connects the behavioral failure observed in
long_recall.py to the internal gate dynamics.

Usage:
    python duplicate_gate_analysis.py --duplicates 5 --layer 0
    python duplicate_gate_analysis.py --sweep-duplicates 0 1 3 5 10
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from common import setup_model, tokenize_text
from long_recall import generate_prompt, format_for_model, NUMBER_WORDS
from decay_gate_analysis import extract_gates, TARGET_LAYERS


def find_apple_positions(token_texts, target_word):
    """
    Find token positions where target_word is assigned a score.
    Returns list of (position, score_word) tuples, where position is the
    index of the score token following "target_word is".
    """
    assignments = []
    score_words = set(NUMBER_WORDS.values())

    for i, tok in enumerate(token_texts):
        if target_word.lower() in tok.strip().lower():
            # Look ahead for "is" then the score token
            for j in range(i + 1, min(i + 4, len(token_texts))):
                if token_texts[j].strip().lower() in score_words:
                    assignments.append((j, token_texts[j].strip().lower()))
                    break

    return assignments


def analyze_single(model, tokenizer, device, num_duplicates, num_distractors,
                   target_word, target_score, seed, layers):
    """
    Analyze gate values at apple assignment positions for a single prompt.
    Returns dict: {layer_idx: (assignments, alpha_at_assignments, beta_at_assignments)}
    where alpha/beta are [num_assignments, num_heads].
    """
    prompt_text, _, expected_word = generate_prompt(
        target_word, target_score, num_distractors,
        num_duplicates=num_duplicates, seed=seed
    )
    formatted = format_for_model(prompt_text, tokenizer)
    input_ids, attention_mask = tokenize_text(formatted, tokenizer, device)

    # Tokenize to find apple positions
    token_texts = [tokenizer.decode(input_ids[0, t:t+1]) for t in range(input_ids.shape[1])]
    assignments = find_apple_positions(token_texts, target_word)

    if not assignments:
        print("Warning: no apple assignments found in prompt")
        return {}, assignments

    print(f"Found {len(assignments)} '{target_word}' assignments:")
    for idx, (pos, score) in enumerate(assignments):
        label = "CORRECT" if score == expected_word else "wrong"
        print(f"  #{idx}: position {pos}, score='{score}' ({label})")

    results = {}
    for layer_idx in layers:
        token_texts_layer, g_matrix, beta_matrix = extract_gates(
            model, tokenizer, input_ids, attention_mask, layer_idx
        )
        if g_matrix is None:
            continue

        alpha_matrix = np.exp(g_matrix)  # [num_heads, num_tokens]
        positions = [pos for pos, _ in assignments]

        alpha_at = alpha_matrix[:, positions].T  # [num_assignments, num_heads]
        beta_at = beta_matrix[:, positions].T    # [num_assignments, num_heads]

        results[layer_idx] = (assignments, alpha_at, beta_at)

    return results, assignments


def plot_single_layer(layer_idx, assignments, alpha_at, beta_at, expected_word,
                      output_dir):
    """Plot alpha and beta at each apple assignment for one layer."""
    os.makedirs(output_dir, exist_ok=True)
    num_assignments, num_heads = alpha_at.shape

    labels = []
    for idx, (pos, score) in enumerate(assignments):
        tag = " (correct)" if score == expected_word else ""
        labels.append(f"#{idx}: {score}{tag}")

    x = np.arange(num_assignments)

    # Mean across heads with std
    alpha_mean = alpha_at.mean(axis=1)
    alpha_std = alpha_at.std(axis=1)
    beta_mean = beta_at.mean(axis=1)
    beta_std = beta_at.std(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, num_assignments * 1.2), 8), sharex=True)

    ax1.bar(x, alpha_mean, yerr=alpha_std, capsize=4, color="steelblue", alpha=0.8)
    ax1.set_ylabel("α (retention)")
    ax1.set_title(f"Layer {layer_idx} — Gate values at target assignments")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    ax2.bar(x, beta_mean, yerr=beta_std, capsize=4, color="coral", alpha=0.8)
    ax2.set_ylabel("β (update)")
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_xlabel("Assignment index")

    plt.tight_layout()
    path = os.path.join(output_dir, f"layer_{layer_idx}_gates.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_per_head_heatmap(layer_idx, assignments, alpha_at, beta_at, expected_word,
                          output_dir):
    """Heatmap of beta at apple positions × heads for one layer."""
    os.makedirs(output_dir, exist_ok=True)
    num_assignments, num_heads = beta_at.shape

    labels = []
    for idx, (pos, score) in enumerate(assignments):
        tag = "*" if score == expected_word else ""
        labels.append(f"{score}{tag}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(4, num_assignments * 0.4)))

    im1 = ax1.imshow(alpha_at, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax1.set_ylabel("Assignment index")
    ax1.set_xlabel("Head")
    ax1.set_title(f"α (retention) — Layer {layer_idx}")
    ax1.set_yticks(np.arange(num_assignments))
    ax1.set_yticklabels(labels)
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(beta_at, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax2.set_ylabel("Assignment index")
    ax2.set_xlabel("Head")
    ax2.set_title(f"β (update) — Layer {layer_idx}")
    ax2.set_yticks(np.arange(num_assignments))
    ax2.set_yticklabels(labels)
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    path = os.path.join(output_dir, f"layer_{layer_idx}_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def sweep_and_plot(model, tokenizer, device, duplicate_counts, num_distractors,
                   target_word, target_score, seed, layers, output_dir):
    """
    For each duplicate count, extract gate values at the last (correct) apple
    assignment and plot how they change as duplicates increase.
    """
    # Collect: {layer_idx: {dup_count: (alpha_mean, alpha_std, beta_mean, beta_std)}}
    sweep_data = {l: {} for l in layers}

    for num_dup in duplicate_counts:
        print(f"\n--- Duplicates: {num_dup} ---")
        results, assignments = analyze_single(
            model, tokenizer, device, num_dup, num_distractors,
            target_word, target_score, seed, layers
        )

        for layer_idx, (assigns, alpha_at, beta_at) in results.items():
            # Last assignment is the correct one (or only one if 0 duplicates)
            last_alpha = alpha_at[-1]  # [num_heads]
            last_beta = beta_at[-1]    # [num_heads]
            sweep_data[layer_idx][num_dup] = (
                last_alpha.mean(), last_alpha.std(),
                last_beta.mean(), last_beta.std(),
            )

    # Plot comparison across duplicate counts for each layer
    os.makedirs(output_dir, exist_ok=True)
    for layer_idx in layers:
        if not sweep_data[layer_idx]:
            continue

        dups = sorted(sweep_data[layer_idx].keys())
        alpha_means = [sweep_data[layer_idx][d][0] for d in dups]
        alpha_stds = [sweep_data[layer_idx][d][1] for d in dups]
        beta_means = [sweep_data[layer_idx][d][2] for d in dups]
        beta_stds = [sweep_data[layer_idx][d][3] for d in dups]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(dups, alpha_means, yerr=alpha_stds, fmt="o-", label="α (retention)",
                     capsize=4, color="steelblue")
        ax.errorbar(dups, beta_means, yerr=beta_stds, fmt="s-", label="β (update)",
                     capsize=4, color="coral")
        ax.set_xlabel("Number of conflicting assignments")
        ax.set_ylabel("Gate value (mean ± std across heads)")
        ax.set_title(f"Layer {layer_idx} — Gate response at correct assignment vs. duplicates")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, f"layer_{layer_idx}_sweep.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")

    # Summary plot: mean across all layers
    all_dups = sorted(set(d for ld in sweep_data.values() for d in ld.keys()))
    summary_alpha = []
    summary_beta = []
    for d in all_dups:
        alphas = [sweep_data[l][d][0] for l in layers if d in sweep_data[l]]
        betas = [sweep_data[l][d][2] for l in layers if d in sweep_data[l]]
        summary_alpha.append(np.mean(alphas))
        summary_beta.append(np.mean(betas))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(all_dups, summary_alpha, "o-", label="α (retention)", color="steelblue", linewidth=2)
    ax.plot(all_dups, summary_beta, "s-", label="β (update)", color="coral", linewidth=2)
    ax.set_xlabel("Number of conflicting assignments")
    ax.set_ylabel("Gate value (mean across all layers & heads)")
    ax.set_title("Gate response at correct assignment vs. duplicates (all layers)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "summary_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate analysis on duplicate recall prompts")
    parser.add_argument("--duplicates", type=int, default=5,
                        help="Number of duplicates for single analysis")
    parser.add_argument("--distance", type=int, default=50,
                        help="Number of distractor lines")
    parser.add_argument("--target-word", type=str, default="apple")
    parser.add_argument("--target-score", type=int, default=5, choices=range(1, 10))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer", type=int, default=None,
                        help="Single layer to analyze (default: all linear attention layers)")
    parser.add_argument("--sweep-duplicates", type=int, nargs="+", default=None,
                        help="Sweep over these duplicate counts (e.g. 0 1 3 5 10)")
    args = parser.parse_args()

    layers = [args.layer] if args.layer is not None else TARGET_LAYERS
    output_dir = "plots/duplicate_gates"

    model, tokenizer, device = setup_model()

    if args.sweep_duplicates is not None:
        sweep_and_plot(
            model, tokenizer, device, args.sweep_duplicates, args.distance,
            args.target_word, args.target_score, args.seed, layers, output_dir
        )
    else:
        expected_word = NUMBER_WORDS[args.target_score]
        results, assignments = analyze_single(
            model, tokenizer, device, args.duplicates, args.distance,
            args.target_word, args.target_score, args.seed, layers
        )

        for layer_idx, (assigns, alpha_at, beta_at) in results.items():
            layer_dir = os.path.join(output_dir, f"layer_{layer_idx}")
            plot_single_layer(layer_idx, assigns, alpha_at, beta_at, expected_word, layer_dir)
            plot_per_head_heatmap(layer_idx, assigns, alpha_at, beta_at, expected_word, layer_dir)
