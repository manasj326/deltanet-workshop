"""
Decay Gate Analysis

Extracts and plots the decay gate alpha_t = exp(g) at each token position across
layers and heads, where g = -exp(A_log) * softplus(a + dt_bias).

alpha_t is the retention factor in the gated delta rule:
  M_t = alpha_t * M_{t-1} + beta_t * (v_t - M_{t-1} @ k_t) @ k_t^T

alpha_t in (0, 1): 1 = keep everything, 0 = forget everything.
beta_t in (0, 1): controls how much of the delta update to write.

Plots:
  - Summary heatmaps: layers x tokens (mean alpha and beta across heads)
  - Per-head heatmaps: heads x tokens for alpha and beta (one per layer)
  - Line plot: mean ± std across heads vs token position (one per layer)
  - Combined alpha/beta line plot per layer
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from common import setup_model, tokenize_prompt, prompt_name

TARGET_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]


def patch_forward_for_gate_capture(attn_module, gate_store):
    """
    Monkey-patch GatedDeltaNet forward to capture both gates at each token.
    Appends g and beta (per-head scalars) to gate_store at each call.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    original_forward = attn_module.forward

    def patched_forward(hidden_states, cache_params=None, cache_position=None,
                        attention_mask=None):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        module = attn_module

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        if cache_params is not None:
            conv_state = cache_params.conv_states[module.layer_idx]
            recurrent_state = cache_params.recurrent_states[module.layer_idx]

        mixed_qkv = module.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = module.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, module.head_v_dim)

        b = module.in_proj_b(hidden_states)
        a = module.in_proj_a(hidden_states)

        if use_precomputed_states:
            mixed_qkv = module.causal_conv1d_update(
                mixed_qkv, conv_state,
                module.conv1d.weight.squeeze(1), module.conv1d.bias,
                module.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (module.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[module.layer_idx] = conv_state
            if module.causal_conv1d_fn is not None:
                mixed_qkv = module.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=module.conv1d.weight.squeeze(1),
                    bias=module.conv1d.bias,
                    activation=module.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(module.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [module.key_dim, module.key_dim, module.value_dim],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, module.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, module.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, module.head_v_dim)

        beta = b.sigmoid()
        g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)

        # Capture both gates: already [num_heads] per token
        g_per_head = g[0, 0].detach().cpu().float().numpy()  # [num_heads]
        beta_per_head = beta[0, 0].detach().cpu().float().numpy()  # [num_heads]
        gate_store["g"].append(g_per_head)
        gate_store["beta"].append(beta_per_head)

        if module.num_v_heads // module.num_k_heads > 1:
            query = query.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)
            key = key.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = module.chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = module.recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.recurrent_states[module.layer_idx] = last_recurrent_state

        core_attn_out = core_attn_out.reshape(-1, module.head_v_dim)
        z = z.reshape(-1, module.head_v_dim)
        core_attn_out = module.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = module.out_proj(core_attn_out)
        return output

    attn_module.forward = patched_forward

    def cleanup():
        attn_module.forward = original_forward

    return cleanup


def extract_gates(model, tokenizer, input_ids, attention_mask, layer_idx):
    """
    Run the model token-by-token for one layer, capturing g and beta at each step.
    Returns (token_texts, g_matrix, beta_matrix) where each is [num_heads, num_tokens].
    """
    layer_module = model.model.layers[layer_idx]
    if not hasattr(layer_module, 'linear_attn'):
        print(f"Layer {layer_idx} is not a linear attention layer, skipping.")
        return None, None, None

    attn_module = layer_module.linear_attn
    gate_store = {"g": [], "beta": []}

    cleanup = patch_forward_for_gate_capture(attn_module, gate_store)

    token_texts = []
    past_key_values = None

    with torch.no_grad():
        for t in range(input_ids.shape[1]):
            current_ids = input_ids[:, t:t+1]
            current_mask = attention_mask[:, :t+1]
            outputs = model(
                input_ids=current_ids,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            token_texts.append(tokenizer.decode(current_ids[0]))

    cleanup()

    if not gate_store["g"]:
        print(f"Layer {layer_idx}: failed to capture gates.")
        return None, None, None

    # Stack: list of [num_heads] -> [num_tokens, num_heads] -> transpose to [num_heads, num_tokens]
    g_matrix = np.stack(gate_store["g"], axis=0).T  # [num_heads, num_tokens]
    beta_matrix = np.stack(gate_store["beta"], axis=0).T  # [num_heads, num_tokens]
    return token_texts, g_matrix, beta_matrix


def save_layer_values(layer_idx, g_matrix, beta_matrix, token_texts, output_dir):
    """
    Dump raw alpha/beta values for one layer.
      - values.csv: wide format, one row per token, columns = head_<h>_alpha, head_<h>_beta
      - values.npz: arrays (alpha [H,T], beta [H,T], tokens [T]) for programmatic use
      - per_head_stats.csv: aggregate stats per head (mean/std/min/max for alpha and beta)
    """
    os.makedirs(output_dir, exist_ok=True)
    alpha = np.exp(g_matrix)
    num_heads, num_tokens = alpha.shape

    csv_path = os.path.join(output_dir, "values.csv")
    with open(csv_path, "w") as f:
        header = ["token_idx", "token"]
        for h in range(num_heads):
            header += [f"head_{h}_alpha", f"head_{h}_beta"]
        f.write(",".join(header) + "\n")
        for t in range(num_tokens):
            tok_repr = token_texts[t].replace(",", "\\,").replace("\n", "\\n")
            row = [str(t), tok_repr]
            for h in range(num_heads):
                row += [f"{alpha[h, t]:.6f}", f"{beta_matrix[h, t]:.6f}"]
            f.write(",".join(row) + "\n")

    # Separate α and β matrices: tokens (rows) × heads (cols), easier to eyeball
    for name, mat in [("alpha", alpha), ("beta", beta_matrix)]:
        path = os.path.join(output_dir, f"{name}.csv")
        with open(path, "w") as f:
            f.write("token_idx,token," + ",".join(f"head_{h}" for h in range(num_heads)) + "\n")
            for t in range(num_tokens):
                tok_repr = token_texts[t].replace(",", "\\,").replace("\n", "\\n")
                row = [str(t), tok_repr] + [f"{mat[h, t]:.6f}" for h in range(num_heads)]
                f.write(",".join(row) + "\n")

    stats_path = os.path.join(output_dir, "per_head_stats.csv")
    with open(stats_path, "w") as f:
        f.write("head,alpha_mean,alpha_std,alpha_min,alpha_max,"
                "beta_mean,beta_std,beta_min,beta_max\n")
        for h in range(num_heads):
            a, b = alpha[h], beta_matrix[h]
            f.write(f"{h},{a.mean():.6f},{a.std():.6f},{a.min():.6f},{a.max():.6f},"
                    f"{b.mean():.6f},{b.std():.6f},{b.min():.6f},{b.max():.6f}\n")

    np.savez(
        os.path.join(output_dir, "values.npz"),
        alpha=alpha, beta=beta_matrix, tokens=np.array(token_texts, dtype=object),
    )


def plot_layer(layer_idx, g_matrix, beta_matrix, token_texts, output_dir):
    """Plot per-head heatmaps, line plots, and combined alpha/beta plot for one layer."""
    os.makedirs(output_dir, exist_ok=True)
    num_heads, num_tokens = g_matrix.shape
    alpha = np.exp(g_matrix)  # retention factor in (0, 1)

    labels = [t.strip() if len(t.strip()) <= 6 else t.strip()[:5] + "…" for t in token_texts]

    # --- Alpha heatmap: heads x tokens ---
    fig, ax = plt.subplots(figsize=(max(14, num_tokens * 0.35), max(5, num_heads * 0.3)))
    im = ax.imshow(alpha, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Head")
    ax.set_title(f"Retention factor α = exp(g) — Layer {layer_idx}")
    ax.set_yticks(np.arange(num_heads))
    if num_tokens <= 60:
        ax.set_xticks(np.arange(num_tokens))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_xticks(np.arange(-0.5, num_tokens, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_heads, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.3, alpha=0.3)
    ax.tick_params(which="minor", length=0)
    plt.colorbar(im, ax=ax, label="α = exp(g)  (1 = retain, 0 = forget)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "alpha_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Beta heatmap: heads x tokens ---
    fig, ax = plt.subplots(figsize=(max(14, num_tokens * 0.35), max(5, num_heads * 0.3)))
    im = ax.imshow(beta_matrix, aspect="auto", cmap="PuBuGn", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Head")
    ax.set_title(f"Update gate β — Layer {layer_idx}")
    ax.set_yticks(np.arange(num_heads))
    if num_tokens <= 60:
        ax.set_xticks(np.arange(num_tokens))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_xticks(np.arange(-0.5, num_tokens, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_heads, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.3, alpha=0.3)
    ax.tick_params(which="minor", length=0)
    plt.colorbar(im, ax=ax, label="β  (1 = full write, 0 = no write)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beta_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Combined alpha/beta line plot ---
    mean_alpha = np.mean(alpha, axis=0)
    std_alpha = np.std(alpha, axis=0)
    mean_beta = np.mean(beta_matrix, axis=0)
    std_beta = np.std(beta_matrix, axis=0)
    x = np.arange(num_tokens)

    fig, ax = plt.subplots(figsize=(max(10, num_tokens * 0.2), 5))
    ax.plot(x, mean_alpha, color="steelblue", linewidth=1.5, label="α (retain)")
    ax.fill_between(x, mean_alpha - std_alpha, mean_alpha + std_alpha,
                     alpha=0.15, color="steelblue")
    ax.plot(x, mean_beta, color="coral", linewidth=1.5, label="β (write)")
    ax.fill_between(x, mean_beta - std_beta, mean_beta + std_beta,
                     alpha=0.15, color="coral")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Gate value")
    ax.set_title(f"α (retain) vs β (write) — Layer {layer_idx}")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.axhline(y=0.0, color="gray", linestyle="-", linewidth=0.5)
    if num_tokens <= 60:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Per-head combined plots ---
    per_head_dir = os.path.join(output_dir, "per_head")
    os.makedirs(per_head_dir, exist_ok=True)
    for h in range(num_heads):
        fig, ax = plt.subplots(figsize=(max(10, num_tokens * 0.2), 5))
        ax.plot(x, alpha[h], color="steelblue", linewidth=1.5, label="α (retain)")
        ax.plot(x, beta_matrix[h], color="coral", linewidth=1.5, label="β (write)")
        ax.set_xlabel("Token position", fontsize=18)
        ax.set_ylabel("Gate value", fontsize=18)
        ax.set_title(f"α vs β — Layer {layer_idx}, Head {h}", fontsize=22)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
        ax.axhline(y=0.0, color="gray", linestyle="-", linewidth=0.5)
        if num_tokens <= 60:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(per_head_dir, f"head_{h}.png"), dpi=150, bbox_inches="tight")
        plt.close()


def plot_gate_per_token(all_data, layers, token_texts, output_path, mode="beta"):
    """Single line plot: mean gate value per token, averaged across all heads and layers."""
    num_tokens = len(token_texts)
    if mode == "alpha":
        per_layer_means = np.stack(
            [np.mean(np.exp(all_data[l]), axis=0) for l in layers if l in all_data], axis=0
        )
        color, ylabel, label = "steelblue", "α (retain)", "mean α"
        title = "Mean α per token (averaged across all heads and layers)"
    else:
        per_layer_means = np.stack(
            [np.mean(all_data[l], axis=0) for l in layers if l in all_data], axis=0
        )
        color, ylabel, label = "coral", "β (write)", "mean β"
        title = "Mean β per token (averaged across all heads and layers)"

    mean_v = per_layer_means.mean(axis=0)
    std_v = per_layer_means.std(axis=0)

    labels = [t.strip() if len(t.strip()) <= 6 else t.strip()[:5] + "…" for t in token_texts]
    x = np.arange(num_tokens)

    fig, ax = plt.subplots(figsize=(max(10, num_tokens * 0.2), 5))
    ax.plot(x, mean_v, color=color, linewidth=1.5, label=label)
    ax.fill_between(x, mean_v - std_v, mean_v + std_v,
                    alpha=0.15, color=color, label="±1 std across layers")
    ax.set_xlabel("Token position")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.axhline(y=0.0, color="gray", linestyle="-", linewidth=0.5)
    if num_tokens <= 60:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary_heatmap(all_data, layers, token_texts, output_path, mode="alpha"):
    """Plot summary heatmap: layers x tokens, mean gate value across heads."""
    num_tokens = len(token_texts)
    num_layers = len(layers)

    matrix = np.zeros((num_layers, num_tokens))
    for i, l in enumerate(layers):
        if l in all_data:
            if mode == "alpha":
                matrix[i] = np.mean(np.exp(all_data[l]), axis=0)
            else:
                matrix[i] = np.mean(all_data[l], axis=0)

    labels = [t.strip() if len(t.strip()) <= 6 else t.strip()[:5] + "…" for t in token_texts]

    if mode == "alpha":
        cmap, title, cbar = "RdYlGn", "Mean α = exp(g) per layer per token", "α (1 = retain, 0 = forget)"
    else:
        cmap, title, cbar = "PuBuGn", "Mean β per layer per token", "β (1 = full write, 0 = no write)"

    fig, ax = plt.subplots(figsize=(max(14, num_tokens * 0.35), max(6, num_layers * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([str(l) for l in layers])
    if num_tokens <= 60:
        ax.set_xticks(range(num_tokens))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_xticks(np.arange(-0.5, num_tokens, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_layers, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.3, alpha=0.3)
    ax.tick_params(which="minor", length=0)
    plt.colorbar(im, ax=ax, label=cbar)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decay gate analysis")
    parser.add_argument("--prompt", type=str, default="prompts/recall.txt",
                        help="Path to prompt file")
    parser.add_argument("--layer", type=int, default=None,
                        help="Single layer index (default: all DeltaNet layers)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific layers to analyze (default: all DeltaNet layers)")
    args = parser.parse_args()

    layers = (
        [args.layer] if args.layer is not None
        else args.layers if args.layers
        else TARGET_LAYERS
    )
    pname = prompt_name(args.prompt)

    model, tokenizer, device = setup_model()
    input_ids, attention_mask = tokenize_prompt(args.prompt, tokenizer, device)

    output_dir = os.path.join("plots", pname, "decay_gate")
    os.makedirs(output_dir, exist_ok=True)

    all_g = {}
    all_beta = {}
    token_texts = None

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        texts, g_matrix, beta_matrix = extract_gates(
            model, tokenizer, input_ids, attention_mask, layer_idx
        )
        if g_matrix is None:
            continue

        if token_texts is None:
            token_texts = texts

        all_g[layer_idx] = g_matrix
        all_beta[layer_idx] = beta_matrix

        # Per-layer plots
        layer_dir = os.path.join(output_dir, f"layer_{layer_idx}")
        plot_layer(layer_idx, g_matrix, beta_matrix, token_texts, layer_dir)
        save_layer_values(layer_idx, g_matrix, beta_matrix, token_texts, layer_dir)
        print(f"  Saved alpha/beta heatmaps, plots, and raw values to {layer_dir}/")

    # Summary heatmaps across all layers
    if all_g and token_texts:
        plot_summary_heatmap(all_g, layers, token_texts,
                             os.path.join(output_dir, "alpha_summary.png"), mode="alpha")
        plot_summary_heatmap(all_beta, layers, token_texts,
                             os.path.join(output_dir, "beta_summary.png"), mode="beta")
        plot_gate_per_token(all_beta, layers, token_texts,
                            os.path.join(output_dir, "beta_per_token.png"), mode="beta")
        plot_gate_per_token(all_g, layers, token_texts,
                            os.path.join(output_dir, "alpha_per_token.png"), mode="alpha")
        print(f"\nSaved summary heatmaps to {output_dir}/")

        all_stats_path = os.path.join(output_dir, "all_layers_stats.csv")
        with open(all_stats_path, "w") as f:
            f.write("layer,head,alpha_mean,alpha_std,alpha_min,alpha_max,"
                    "beta_mean,beta_std,beta_min,beta_max\n")
            for layer_idx in layers:
                if layer_idx not in all_g:
                    continue
                alpha_l = np.exp(all_g[layer_idx])
                beta_l = all_beta[layer_idx]
                for h in range(alpha_l.shape[0]):
                    a, b = alpha_l[h], beta_l[h]
                    f.write(f"{layer_idx},{h},"
                            f"{a.mean():.6f},{a.std():.6f},{a.min():.6f},{a.max():.6f},"
                            f"{b.mean():.6f},{b.std():.6f},{b.min():.6f},{b.max():.6f}\n")
        print(f"Saved combined per-head stats to {all_stats_path}")

        all_values_path = os.path.join(output_dir, "all_layers_values.csv")
        with open(all_values_path, "w") as f:
            f.write("layer,head,token_idx,token,alpha,beta\n")
            for layer_idx in layers:
                if layer_idx not in all_g:
                    continue
                alpha_l = np.exp(all_g[layer_idx])
                beta_l = all_beta[layer_idx]
                num_heads, num_tokens = alpha_l.shape
                for h in range(num_heads):
                    for t in range(num_tokens):
                        tok_repr = token_texts[t].replace(",", "\\,").replace("\n", "\\n")
                        f.write(f"{layer_idx},{h},{t},{tok_repr},"
                                f"{alpha_l[h, t]:.6f},{beta_l[h, t]:.6f}\n")
        print(f"Saved combined per-token values to {all_values_path}")

        all_values_npz = os.path.join(output_dir, "all_layers_values.npz")
        sorted_layers = [l for l in layers if l in all_g]
        alpha_stack = np.stack([np.exp(all_g[l]) for l in sorted_layers], axis=0)
        beta_stack = np.stack([all_beta[l] for l in sorted_layers], axis=0)
        np.savez(
            all_values_npz,
            alpha=alpha_stack, beta=beta_stack,
            layers=np.array(sorted_layers),
            tokens=np.array(token_texts, dtype=object),
        )
        print(f"Saved combined arrays to {all_values_npz}")

    print(f"\nAll plots saved to {output_dir}/")
