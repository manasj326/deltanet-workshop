"""
Causal Tracing for Update Tasks, with Joint State Patching

Runs both a per-layer causal trace and two joint interventions that patch
*all* linear-attn layers' recurrent states at once. The joint patch
collapses the single-layer blind spot: if the binding is redundantly
distributed across LA layers, joint patching can recover even when no
single-layer patch does.

Prompt pair tests the delta-rule's natural use case — overwriting an
earlier binding with a later one:
  Clean   : "... cat is 3. ... cat is now 7. What is the score of cat?"
  Corrupt : "... cat is 3. ... cat is now 9. What is the score of cat?"

Outputs:
  residual.png  — per-layer residual heatmap   (L x T)
  state.png     — per-layer recurrent-state heatmap (L x T, SA rows NaN)
  joint.png     — single-timestep and cumulative joint-patch curves
  arrays.npz    — all numeric outputs plus positions of interest
"""

import os
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from common import setup_model, tokenize_text, MODEL_NAME


def model_label(name=MODEL_NAME):
    # "Qwen/Qwen3.5-0.8B" -> "Qwen3.5 0.8B"
    return name.split("/")[-1].replace("-", " ")


DEFAULT_CLEAN = (
    "The score of cat is 3. The score of cat is now 7. "
    "What is the score of cat? Answer with just the number."
)
DEFAULT_CORRUPT = (
    "The score of cat is 3. The score of cat is now 9. "
    "What is the score of cat? Answer with just the number."
)


def format_prompt(text, tokenizer):
    if "Base" in MODEL_NAME or not tokenizer.chat_template:
        return text
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_pair(clean_text, corrupt_text, tokenizer, device):
    clean_ids, clean_mask = tokenize_text(
        format_prompt(clean_text, tokenizer), tokenizer, device
    )
    corrupt_ids, corrupt_mask = tokenize_text(
        format_prompt(corrupt_text, tokenizer), tokenizer, device
    )
    if clean_ids.shape != corrupt_ids.shape:
        raise ValueError(
            f"Tokenized shapes differ: {clean_ids.shape} vs {corrupt_ids.shape}."
        )
    diff = (clean_ids != corrupt_ids).nonzero(as_tuple=False)
    if diff.shape[0] != 1:
        raise ValueError(
            f"Expected exactly one differing token, found {diff.shape[0]}."
        )
    diff_pos = int(diff[0, 1])
    return {
        "clean_ids": clean_ids,
        "clean_mask": clean_mask,
        "corrupt_ids": corrupt_ids,
        "corrupt_mask": corrupt_mask,
        "diff_pos": diff_pos,
        "clean_tok": int(clean_ids[0, diff_pos]),
        "corrupt_tok": int(corrupt_ids[0, diff_pos]),
    }


def find_positions_of_interest(pair, tokenizer):
    """Derive old_val (first standalone digit token), new_val (=diff_pos),
    query_cat (last 'cat' in the sequence), end (last position)."""
    token_texts = [tokenizer.decode(t) for t in pair["corrupt_ids"][0].tolist()]
    T = len(token_texts)

    old_val = None
    for i, tok in enumerate(token_texts):
        stripped = tok.strip()
        if stripped.isdigit() and len(stripped) == 1 and i != pair["diff_pos"]:
            old_val = i
            break

    query_cat = None
    for i, tok in enumerate(token_texts):
        if tok.strip() == "cat":
            query_cat = i

    return {
        "old_val": old_val,
        "new_val": pair["diff_pos"],
        "query_cat": query_cat,
        "end": T - 1,
        "token_texts": token_texts,
    }


def logit_diff(logits, clean_tok, corrupt_tok):
    last = logits[0, -1].float()
    return float(last[clean_tok] - last[corrupt_tok])


def run_clean_caches(model, clean_ids, clean_mask):
    with torch.no_grad():
        outputs = model(
            input_ids=clean_ids,
            attention_mask=clean_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden = [h.detach() for h in outputs.hidden_states]

    past_key_values = None
    T = clean_ids.shape[1]
    states = None
    with torch.no_grad():
        for t in range(T):
            out = model(
                input_ids=clean_ids[:, t:t + 1],
                attention_mask=clean_mask[:, :t + 1],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            rs = past_key_values.recurrent_states
            if states is None:
                states = [[] for _ in range(len(rs))]
            for ell, s in enumerate(rs):
                states[ell].append(s.detach().clone() if s is not None else None)

    return hidden, states


def residual_patch(model, corrupt_ids, corrupt_mask, clean_hidden,
                   ell, t, clean_tok, corrupt_tok):
    replacement = clean_hidden[ell + 1][:, t:t + 1]
    layer_module = model.model.layers[ell]

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0].clone()
            h[:, t:t + 1] = replacement
            return (h,) + output[1:]
        h = output.clone()
        h[:, t:t + 1] = replacement
        return h

    handle = layer_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(
                input_ids=corrupt_ids,
                attention_mask=corrupt_mask,
                use_cache=False,
            )
    finally:
        handle.remove()
    return logit_diff(out.logits, clean_tok, corrupt_tok)


def state_patch(model, corrupt_ids, corrupt_mask, clean_states,
                ell, target_t, clean_tok, corrupt_tok):
    past_key_values = None
    T = corrupt_ids.shape[1]
    final_logits = None
    with torch.no_grad():
        for t in range(T):
            out = model(
                input_ids=corrupt_ids[:, t:t + 1],
                attention_mask=corrupt_mask[:, :t + 1],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            if t == target_t:
                cs = clean_states[ell][t]
                if cs is not None:
                    past_key_values.recurrent_states[ell].copy_(cs)
            final_logits = out.logits
    return logit_diff(final_logits, clean_tok, corrupt_tok)


def joint_state_patch(model, corrupt_ids, corrupt_mask, clean_states,
                      target_t, clean_tok, corrupt_tok):
    """Overwrite *all* LA layers' recurrent states at step target_t."""
    past_key_values = None
    T = corrupt_ids.shape[1]
    final_logits = None
    with torch.no_grad():
        for t in range(T):
            out = model(
                input_ids=corrupt_ids[:, t:t + 1],
                attention_mask=corrupt_mask[:, :t + 1],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            if t == target_t:
                for ell, cs_list in enumerate(clean_states):
                    cs = cs_list[t]
                    if cs is not None:
                        past_key_values.recurrent_states[ell].copy_(cs)
            final_logits = out.logits
    return logit_diff(final_logits, clean_tok, corrupt_tok)


def cumulative_state_patch(model, corrupt_ids, corrupt_mask, clean_states,
                           until_t, clean_tok, corrupt_tok):
    """Overwrite *all* LA layers' recurrent states at every step t <= until_t."""
    past_key_values = None
    T = corrupt_ids.shape[1]
    final_logits = None
    with torch.no_grad():
        for t in range(T):
            out = model(
                input_ids=corrupt_ids[:, t:t + 1],
                attention_mask=corrupt_mask[:, :t + 1],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            if t <= until_t:
                for ell, cs_list in enumerate(clean_states):
                    cs = cs_list[t]
                    if cs is not None:
                        past_key_values.recurrent_states[ell].copy_(cs)
            final_logits = out.logits
    return logit_diff(final_logits, clean_tok, corrupt_tok)


def per_layer_sweep(model, pair, clean_hidden, clean_states, corrupt_ld,
                    denom, layers, skip_state=False):
    L_total = len(model.model.layers)
    T = pair["clean_ids"].shape[1]
    if layers is None:
        layers = list(range(L_total))

    residual = np.full((L_total, T), np.nan)
    state = np.full((L_total, T), np.nan)
    residual_raw = np.full((L_total, T), np.nan)
    state_raw = np.full((L_total, T), np.nan)

    total_start = time.time()
    for i, ell in enumerate(layers):
        layer_start = time.time()
        has_state = clean_states[ell][0] is not None
        for t in range(T):
            ld_r = residual_patch(
                model, pair["corrupt_ids"], pair["corrupt_mask"],
                clean_hidden, ell, t,
                pair["clean_tok"], pair["corrupt_tok"],
            )
            residual_raw[ell, t] = ld_r
            residual[ell, t] = (ld_r - corrupt_ld) / denom

            if has_state and not skip_state:
                ld_s = state_patch(
                    model, pair["corrupt_ids"], pair["corrupt_mask"],
                    clean_states, ell, t,
                    pair["clean_tok"], pair["corrupt_tok"],
                )
                state_raw[ell, t] = ld_s
                state[ell, t] = (ld_s - corrupt_ld) / denom
        dt = time.time() - layer_start
        print(f"  layer {ell:2d}  ({i + 1}/{len(layers)})  {dt:5.1f}s", flush=True)

    print(f"  per-layer total: {time.time() - total_start:.1f}s", flush=True)
    return residual, state, residual_raw, state_raw


def joint_sweep(model, pair, clean_states, corrupt_ld, denom):
    T = pair["clean_ids"].shape[1]
    single = np.full(T, np.nan)
    cumulative = np.full(T, np.nan)
    single_raw = np.full(T, np.nan)
    cumulative_raw = np.full(T, np.nan)

    start = time.time()
    for t in range(T):
        ld_s = joint_state_patch(
            model, pair["corrupt_ids"], pair["corrupt_mask"],
            clean_states, t,
            pair["clean_tok"], pair["corrupt_tok"],
        )
        single_raw[t] = ld_s
        single[t] = (ld_s - corrupt_ld) / denom

        ld_c = cumulative_state_patch(
            model, pair["corrupt_ids"], pair["corrupt_mask"],
            clean_states, t,
            pair["clean_tok"], pair["corrupt_tok"],
        )
        cumulative_raw[t] = ld_c
        cumulative[t] = (ld_c - corrupt_ld) / denom
    print(f"  joint total: {time.time() - start:.1f}s", flush=True)
    return single, cumulative, single_raw, cumulative_raw


def plot_heatmap(data, token_texts, title, out_path, positions):
    fig, ax = plt.subplots(figsize=(max(10, 0.28 * data.shape[1]), 7))
    im = ax.imshow(
        data, aspect="auto", cmap="RdBu_r",
        vmin=-0.2, vmax=1.2, origin="lower",
    )
    ax.set_xlabel("Token position", fontsize=22)
    ax.set_ylabel("Layer", fontsize=22)
    ax.set_title(title, fontsize=22, pad=28)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_xticks(range(len(token_texts)))
    ax.set_xticklabels(
        [t.replace("\n", "\\n").strip() or "_" for t in token_texts],
        rotation=90, fontsize=15,
    )
    marker_labels = {"old_val": "old", "new_val": "new",
                     "query_cat": "qry", "end": "end"}
    for name, pos in positions.items():
        label = marker_labels.get(name)
        if label is None or pos is None:
            continue
        ax.axvline(pos, color="green", linewidth=0.8, alpha=0.7)
        ax.text(pos, data.shape[0] + 0.3, label,
                ha="center", va="bottom", fontsize=14, color="green")
    cbar = fig.colorbar(im, ax=ax, label="Recovery (0 = corrupt, 1 = clean)")
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Recovery (0 = corrupt, 1 = clean)", fontsize=22)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def plot_joint_curves(single, cumulative, token_texts, positions, out_path):
    T = len(single)
    fig, ax = plt.subplots(figsize=(max(10, 0.28 * T), 5))
    xs = np.arange(T)
    ax.plot(xs, single, "-o", label="single-timestep joint", markersize=4)
    ax.plot(xs, cumulative, "-o", label="cumulative joint (t ≤ t*)", markersize=4)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axhline(1, color="gray", linewidth=0.5, linestyle="--")

    marker_labels = {"old_val": "old", "new_val": "new",
                     "query_cat": "qry", "end": "end"}
    for name, pos in positions.items():
        label = marker_labels.get(name)
        if label is None or pos is None:
            continue
        ax.axvline(pos, color="green", linewidth=0.7, alpha=0.7)
        ax.text(pos, 1.15, label, ha="center", va="bottom",
                fontsize=14, color="green")

    ax.set_xlabel("Target position t*", fontsize=18)
    ax.set_ylabel("Recovery (0 = corrupt, 1 = clean)", fontsize=22)
    ax.set_title(f"Joint state patching (all LA layers) — update task — {model_label()}",
                 fontsize=20, pad=28)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [t.replace("\n", "\\n").strip() or "_" for t in token_texts],
        rotation=90, fontsize=15,
    )
    ax.set_ylim(-0.3, 1.3)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=str, default=DEFAULT_CLEAN)
    parser.add_argument("--corrupt", type=str, default=DEFAULT_CORRUPT)
    parser.add_argument("--out-dir", type=str,
                        default="plots/update/causal_trace_joint")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices for per-layer sweep.")
    parser.add_argument("--joint-only", action="store_true",
                        help="Skip per-layer residual/state sweep.")
    parser.add_argument("--skip-joint", action="store_true",
                        help="Skip joint state patching.")
    parser.add_argument("--skip-state", action="store_true",
                        help="Skip per-layer STATE patching (residual only).")
    parser.add_argument("--replot-only", action="store_true",
                        help="Load arrays.npz from --out-dir and regenerate PNGs only "
                             "(skip model + sweeps).")
    args = parser.parse_args()

    if args.replot_only:
        npz_path = os.path.join(args.out_dir, "arrays.npz")
        print(f"Replotting from {npz_path}", flush=True)
        data = np.load(npz_path, allow_pickle=True)
        token_texts = list(data["token_texts"])
        positions = {
            "old_val": int(data["old_val_pos"]) if int(data["old_val_pos"]) >= 0 else None,
            "new_val": int(data["new_val_pos"]),
            "query_cat": int(data["query_cat_pos"]) if int(data["query_cat_pos"]) >= 0 else None,
            "end": int(data["end_pos"]),
        }
        if "residual" in data.files:
            plot_heatmap(
                data["residual"], token_texts,
                f"Causal trace — residual stream (update) — {model_label()}",
                os.path.join(args.out_dir, "residual.png"), positions,
            )
        if "state" in data.files:
            plot_heatmap(
                data["state"], token_texts,
                f"Causal trace — recurrent state (update) — {model_label()}",
                os.path.join(args.out_dir, "state.png"), positions,
            )
        if "joint_single" in data.files:
            plot_joint_curves(
                data["joint_single"], data["joint_cumulative"],
                token_texts, positions,
                os.path.join(args.out_dir, "joint.png"),
            )
        return

    model, tokenizer, device = setup_model()
    pair = build_pair(args.clean, args.corrupt, tokenizer, device)
    positions = find_positions_of_interest(pair, tokenizer)
    token_texts = positions["token_texts"]

    print(f"Sequence length: {len(token_texts)}", flush=True)
    print(f"Positions of interest:", flush=True)
    for key in ("old_val", "new_val", "query_cat", "end"):
        pos = positions[key]
        if pos is None:
            print(f"  {key:10s}: not found", flush=True)
        else:
            print(f"  {key:10s}: pos {pos:3d}  tok={token_texts[pos]!r}", flush=True)
    print(f"Differing tokens: clean={tokenizer.decode(pair['clean_tok'])!r} "
          f"corrupt={tokenizer.decode(pair['corrupt_tok'])!r}", flush=True)

    with torch.no_grad():
        clean_out = model(
            input_ids=pair["clean_ids"], attention_mask=pair["clean_mask"]
        )
        corrupt_out = model(
            input_ids=pair["corrupt_ids"], attention_mask=pair["corrupt_mask"]
        )
    clean_ld = logit_diff(clean_out.logits, pair["clean_tok"], pair["corrupt_tok"])
    corrupt_ld = logit_diff(corrupt_out.logits, pair["clean_tok"], pair["corrupt_tok"])
    print(f"Baseline logit diff:  clean {clean_ld:+.3f}   "
          f"corrupt {corrupt_ld:+.3f}", flush=True)
    denom = clean_ld - corrupt_ld
    if denom <= 0:
        raise RuntimeError(
            "Clean prompt does not prefer clean value over corrupt value. "
            "Model isn't respecting the update — experiment is invalid."
        )

    print("Capturing clean residual + recurrent state caches...", flush=True)
    clean_hidden, clean_states = run_clean_caches(
        model, pair["clean_ids"], pair["clean_mask"]
    )

    os.makedirs(args.out_dir, exist_ok=True)

    residual = state = residual_raw = state_raw = None
    if not args.joint_only:
        layers = None
        if args.layers is not None:
            layers = [int(x) for x in args.layers.split(",")]
        print("Per-layer sweep...", flush=True)
        residual, state, residual_raw, state_raw = per_layer_sweep(
            model, pair, clean_hidden, clean_states, corrupt_ld, denom, layers,
            skip_state=args.skip_state,
        )
        plot_heatmap(
            residual, token_texts,
            f"Causal trace — residual stream (update) — {model_label()}",
            os.path.join(args.out_dir, "residual.png"),
            positions,
        )
        if not args.skip_state:
            plot_heatmap(
                state, token_texts,
                f"Causal trace — recurrent state (update) — {model_label()}",
                os.path.join(args.out_dir, "state.png"),
                positions,
            )

    single = cumulative = single_raw = cumulative_raw = None
    if not args.skip_joint:
        print("Joint state patching sweep...", flush=True)
        single, cumulative, single_raw, cumulative_raw = joint_sweep(
            model, pair, clean_states, corrupt_ld, denom
        )
        plot_joint_curves(
            single, cumulative, token_texts, positions,
            os.path.join(args.out_dir, "joint.png"),
        )

    save_kwargs = {
        "clean_ld": clean_ld,
        "corrupt_ld": corrupt_ld,
        "token_texts": np.array(token_texts, dtype=object),
        "diff_pos": pair["diff_pos"],
        "old_val_pos": -1 if positions["old_val"] is None else positions["old_val"],
        "new_val_pos": positions["new_val"],
        "query_cat_pos": -1 if positions["query_cat"] is None else positions["query_cat"],
        "end_pos": positions["end"],
    }
    if residual is not None:
        save_kwargs.update({
            "residual": residual,
            "state": state,
            "residual_raw": residual_raw,
            "state_raw": state_raw,
        })
    if single is not None:
        save_kwargs.update({
            "joint_single": single,
            "joint_cumulative": cumulative,
            "joint_single_raw": single_raw,
            "joint_cumulative_raw": cumulative_raw,
        })
    np.savez(os.path.join(args.out_dir, "arrays.npz"), **save_kwargs)
    print(f"Saved arrays to {os.path.join(args.out_dir, 'arrays.npz')}", flush=True)


if __name__ == "__main__":
    main()
