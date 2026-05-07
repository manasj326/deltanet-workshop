"""
Gate Intervention on Duplicate Recall

Causally tests whether forcing better gate values (higher beta / lower alpha)
at apple assignment positions recovers recall accuracy in duplicate prompts.

Usage:
    python gate_intervention.py --sweep-duplicates 0 1 3 5 10
    python gate_intervention.py --sweep-duplicates 0 3 5 10 --trials 3 --layer 0
"""

import os
import math
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from common import setup_model, tokenize_text
from long_recall import generate_prompt, format_for_model, check_answer, NUMBER_WORDS
from duplicate_gate_analysis import find_apple_positions
from decay_gate_analysis import TARGET_LAYERS


def patch_forward_for_gate_intervention(attn_module, target_positions,
                                         forced_beta=None, forced_alpha=None,
                                         debug_log=None):
    """
    Monkey-patch GatedDeltaNet forward to override gate values at target positions.

    target_positions: set of token indices to intervene on
    forced_beta: if not None, override beta to this value at target positions
    forced_alpha: if not None, override alpha (via g = log(alpha)) at target positions
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    original_forward = attn_module.forward
    # Track the absolute token position across forward calls
    pos_tracker = {"offset": 0}

    # Precompute forced g from alpha: g = log(alpha), but clamp to avoid log(0)
    forced_g = None
    if forced_alpha is not None:
        forced_g = math.log(max(forced_alpha, 1e-10))

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

        # --- Intervention: override gates at target token positions ---
        # beta/g shape: [batch, seq_len, num_heads]
        # Find which positions in this chunk overlap with target_positions
        offset = pos_tracker["offset"]
        for seq_idx in range(seq_len):
            abs_pos = offset + seq_idx
            if abs_pos in target_positions:
                if debug_log is not None:
                    debug_log.append(f"  Layer {module.layer_idx}: intervening at pos {abs_pos} "
                                     f"(beta={forced_beta}, alpha={forced_alpha})")
                if forced_beta is not None:
                    beta[:, seq_idx, :] = forced_beta
                if forced_g is not None:
                    g[:, seq_idx, :] = forced_g
        pos_tracker["offset"] += seq_len

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


def apply_intervention(model, layers, target_positions, forced_beta, forced_alpha,
                       debug_log=None):
    """Patch all target layers. Returns cleanup function."""
    cleanups = []
    for layer_idx in layers:
        layer_module = model.model.layers[layer_idx]
        if not hasattr(layer_module, 'linear_attn'):
            continue
        attn_module = layer_module.linear_attn
        cleanup = patch_forward_for_gate_intervention(
            attn_module, target_positions, forced_beta, forced_alpha,
            debug_log=debug_log
        )
        cleanups.append(cleanup)

    def cleanup_all():
        for c in cleanups:
            c()

    return cleanup_all


def build_position_overrides(assignments, expected_word,
                              correct_beta, correct_alpha,
                              wrong_beta, wrong_alpha):
    """Build a dict mapping token position -> (forced_beta, forced_alpha).

    Merges correct and wrong interventions into a single lookup so we only
    need one patch per layer (avoids stacked monkey-patching issues).
    """
    overrides = {}
    if assignments:
        correct_pos = assignments[-1][0]
        if correct_beta is not None or correct_alpha is not None:
            overrides[correct_pos] = (correct_beta, correct_alpha)

        for pos, score in assignments[:-1]:
            if score != expected_word and (wrong_beta is not None or wrong_alpha is not None):
                overrides[pos] = (wrong_beta, wrong_alpha)

    return overrides


def patch_forward_with_overrides(attn_module, overrides, debug_log=None):
    """Monkey-patch forward with per-position gate overrides.

    overrides: dict mapping token position -> (forced_beta, forced_alpha)
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    original_forward = attn_module.forward
    pos_tracker = {"offset": 0}

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

        offset = pos_tracker["offset"]
        for seq_idx in range(seq_len):
            abs_pos = offset + seq_idx
            if abs_pos in overrides:
                fb, fa = overrides[abs_pos]
                if debug_log is not None:
                    debug_log.append(f"  Layer {module.layer_idx}: pos {abs_pos} "
                                     f"(beta={fb}, alpha={fa})")
                if fb is not None:
                    beta[:, seq_idx, :] = fb
                if fa is not None:
                    forced_g = math.log(max(fa, 1e-10))
                    g[:, seq_idx, :] = forced_g
        pos_tracker["offset"] += seq_len

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


def patch_scale_beta(attn_module, scale, mode="post"):
    """Monkey-patch GatedDeltaNet forward to scale beta at every token.

    mode="post": beta_new = clamp(beta * scale, 0, 1)       -- multiplicative on the sigmoid output
    mode="logit": beta_new = sigmoid(scale * b)              -- scale the pre-sigmoid projection
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

        g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)

        if mode == "logit":
            # Scale the pre-sigmoid projection, sharpening the model's own decisions
            beta = (scale * b).sigmoid()
        else:
            beta = b.sigmoid()
            beta = (beta * scale).clamp(0.0, 1.0)

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


def apply_beta_scale(model, layers, scale, mode="post"):
    """Apply beta scaling to the specified DeltaNet layers.

    mode="post": multiplies beta after sigmoid and clamps to [0,1]
    mode="logit": multiplies b before sigmoid (sharpens model's decisions)
    """
    cleanups = []
    for layer_idx in layers:
        layer_module = model.model.layers[layer_idx]
        if not hasattr(layer_module, 'linear_attn'):
            continue
        cleanups.append(patch_scale_beta(layer_module.linear_attn, scale, mode=mode))

    def cleanup_all():
        for c in cleanups:
            c()

    return cleanup_all


def ablate_standard_attention(model):
    """Register hooks that zero the output of non-DeltaNet self_attn modules.

    Returns a cleanup function that removes the hooks.
    """
    def zero_attn_hook(module, inputs, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)

    handles = []
    ablated = []
    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'linear_attn'):
            continue  # skip DeltaNet layers
        if hasattr(layer, 'self_attn'):
            handles.append(layer.self_attn.register_forward_hook(zero_attn_hook))
            ablated.append(idx)

    def cleanup():
        for h in handles:
            h.remove()

    return cleanup, ablated


def apply_overrides(model, layers, overrides, debug_log=None):
    """Patch all target layers with per-position overrides. Returns cleanup function."""
    cleanups = []
    for layer_idx in layers:
        layer_module = model.model.layers[layer_idx]
        if not hasattr(layer_module, 'linear_attn'):
            continue
        attn_module = layer_module.linear_attn
        cleanup = patch_forward_with_overrides(attn_module, overrides, debug_log=debug_log)
        cleanups.append(cleanup)

    def cleanup_all():
        for c in cleanups:
            c()

    return cleanup_all


def run_single_trial(model, tokenizer, device, prompt_text, target_word, target_score,
                     layers, forced_beta, forced_alpha, max_new_tokens,
                     wrong_beta=None, wrong_alpha=None, ablate_std_attn=False,
                     beta_scale=None, beta_logit_scale=None):
    """Run one trial with optional gate intervention. Returns (response, is_correct).

    forced_beta/forced_alpha: applied at the last (correct) assignment
    wrong_beta/wrong_alpha: applied at all wrong assignments
    beta_scale: if set, multiply beta by this scalar globally across all DeltaNet layers
    """
    formatted = format_for_model(prompt_text, tokenizer)
    input_ids, attention_mask = tokenize_text(formatted, tokenizer, device)

    token_texts = [tokenizer.decode(input_ids[0, t:t+1]) for t in range(input_ids.shape[1])]
    assignments = find_apple_positions(token_texts, target_word)
    expected_word = NUMBER_WORDS[target_score]

    overrides = build_position_overrides(
        assignments, expected_word,
        forced_beta, forced_alpha,
        wrong_beta, wrong_alpha
    )

    cleanup = None
    if overrides:
        cleanup = apply_overrides(model, layers, overrides)

    scale_cleanup = None
    if beta_scale is not None:
        scale_cleanup = apply_beta_scale(model, layers, beta_scale, mode="post")

    logit_scale_cleanup = None
    if beta_logit_scale is not None:
        logit_scale_cleanup = apply_beta_scale(model, layers, beta_logit_scale, mode="logit")

    ablate_cleanup = None
    if ablate_std_attn:
        ablate_cleanup, _ = ablate_standard_attention(model)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    new_tokens = outputs[0, input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if cleanup:
        cleanup()
    if scale_cleanup:
        scale_cleanup()
    if logit_scale_cleanup:
        logit_scale_cleanup()
    if ablate_cleanup:
        ablate_cleanup()

    expected_num = str(target_score)
    is_correct = check_answer(response, expected_num, expected_word)

    return response.strip(), is_correct


def run_intervention_sweep(model, tokenizer, device, duplicate_counts, num_distractors,
                           target_word, target_score, layers, forced_beta, forced_alpha,
                           num_trials, max_new_tokens, wrong_beta_sweep=None,
                           ablate_std_attn=False, beta_scale_sweep=None,
                           beta_logit_scale_sweep=None, minimal=False):
    """Run sweep over duplicate counts with multiple conditions.

    Each condition is: (correct_beta, correct_alpha, wrong_beta, wrong_alpha, beta_scale, beta_logit_scale)
    """
    # Build conditions: name -> (cb, ca, wb, wa, beta_scale, beta_logit_scale)
    conditions = {"baseline": (None, None, None, None, None, None)}
    if not minimal:
        if forced_beta is not None:
            conditions[f"correct:b={forced_beta}"] = (forced_beta, None, None, None, None, None)
        alpha_list = forced_alpha if isinstance(forced_alpha, list) else [forced_alpha] if forced_alpha is not None else []
        for a in alpha_list:
            conditions[f"correct:a={a}"] = (None, a, None, None, None, None)
        if forced_beta is not None:
            for a in alpha_list:
                conditions[f"correct:b={forced_beta}+a={a}"] = (forced_beta, a, None, None, None, None)

        # Wrong-position conditions
        conditions["wrong:b=0"] = (None, None, 0.0, None, None, None)
        conditions["wrong:b=0 + correct:b=1"] = (1.0, None, 0.0, None, None, None)

    # Wrong beta sweep if requested
    if wrong_beta_sweep is not None:
        for wb in wrong_beta_sweep:
            conditions[f"wrong:b={wb}"] = (None, None, wb, None, None, None)

    # Global beta scaling sweep if requested
    if beta_scale_sweep is not None:
        for bs in beta_scale_sweep:
            conditions[f"beta*={bs}"] = (None, None, None, None, bs, None)

    # Logit scaling sweep if requested
    if beta_logit_scale_sweep is not None:
        for bls in beta_logit_scale_sweep:
            conditions[f"logit*={bls}"] = (None, None, None, None, None, bls)

    # results[condition][dup_count] = accuracy
    results = {name: {} for name in conditions}
    trial_log = []

    for num_dup in duplicate_counts:
        print(f"\n{'='*50}")
        print(f"Duplicates: {num_dup}")
        print(f"{'='*50}")

        for cond_name, (cb, ca, wb, wa, bs, bls) in conditions.items():
            correct_count = 0
            print(f"\n  Condition: {cond_name}")

            for trial in range(num_trials):
                prompt_text, _, _ = generate_prompt(
                    target_word, target_score, num_distractors,
                    num_duplicates=num_dup, seed=trial
                )

                response, is_correct = run_single_trial(
                    model, tokenizer, device, prompt_text, target_word, target_score,
                    layers, cb, ca, max_new_tokens,
                    wrong_beta=wb, wrong_alpha=wa,
                    ablate_std_attn=ablate_std_attn,
                    beta_scale=bs, beta_logit_scale=bls,
                )

                if is_correct:
                    correct_count += 1

                trial_log.append({
                    "duplicates": num_dup,
                    "condition": cond_name,
                    "trial": trial,
                    "response": response,
                    "correct": is_correct,
                })

                print(f"    Trial {trial}: correct={is_correct}, response='{response[:60]}'")

            accuracy = correct_count / num_trials
            results[cond_name][num_dup] = accuracy
            print(f"    Accuracy: {accuracy:.0%}")

    return results, trial_log


def load_results_from_log(log_path):
    """Parse the summary section of a results.txt log to recover {condition: {dup: acc}}.

    Used by --from-cache to re-plot without re-running the sweep.
    """
    import re
    results = {}
    current = None
    with open(log_path) as f:
        for line in f:
            stripped = line.rstrip("\n")
            # Condition header line: "<name>:" with no leading whitespace
            if stripped and not stripped.startswith(" ") and stripped.endswith(":"):
                name = stripped[:-1].strip()
                # Skip the trial-table header (no real condition ends with these)
                if name and name not in ("dups", "condition"):
                    current = name
                    results[current] = {}
                continue
            m = re.match(r"\s+Duplicates\s+(-?\d+):\s+(\d+)%", stripped)
            if m and current is not None:
                results[current][int(m.group(1))] = int(m.group(2)) / 100.0
    return results


def plot_intervention_sweep(results, output_dir="plots/gate_intervention"):
    """Grouped bar chart: accuracy by duplicate count, one bar per condition."""
    os.makedirs(output_dir, exist_ok=True)

    conditions = list(results.keys())
    dup_counts = sorted(set(d for r in results.values() for d in r.keys()))

    x = np.arange(len(dup_counts))
    width = 0.8 / len(conditions)
    colors = ["tab:gray", "tab:orange", "tab:blue", "tab:green", "tab:red",
              "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, cond in enumerate(conditions):
        accuracies = [results[cond].get(d, 0) for d in dup_counts]
        ax.bar(x + i * width, accuracies, width, label=cond,
               color=colors[i % len(colors)], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x + width * (len(conditions) - 1) / 2)
    ax.set_xticklabels([str(d) for d in dup_counts], fontsize=26)
    ax.tick_params(axis="y", labelsize=24)
    ax.set_xlabel("Number of conflicting assignments", fontsize=28)
    ax.set_ylabel("Recall accuracy", fontsize=28)
    ax.set_title("Gate Intervention: Accuracy vs. Duplicates", fontsize=30)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1/9, color="red", linestyle="--", alpha=0.3, label="Random (1/9)")
    ax.legend(fontsize=18, loc="center left", bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "intervention_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {path}")


def run_grid(model, tokenizer, device, alpha_values, beta_values, num_duplicates,
             num_distractors, target_word, target_score, layers, num_trials, max_new_tokens):
    """Run alpha x beta grid at a fixed duplicate count. Returns (grid, baseline_accuracy)."""
    # Run baseline first (no intervention)
    print("\n  Baseline (no intervention)")
    baseline_correct = 0
    for trial in range(num_trials):
        prompt_text, _, _ = generate_prompt(
            target_word, target_score, num_distractors,
            num_duplicates=num_duplicates, seed=trial
        )
        response, is_correct = run_single_trial(
            model, tokenizer, device, prompt_text, target_word, target_score,
            layers, None, None, max_new_tokens
        )
        if is_correct:
            baseline_correct += 1
        print(f"    Trial {trial}: correct={is_correct}, response='{response[:40]}'")
    baseline_accuracy = baseline_correct / num_trials
    print(f"    Baseline accuracy: {baseline_accuracy:.0%}")

    grid = np.zeros((len(alpha_values), len(beta_values)))

    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            correct_count = 0
            print(f"\n  alpha={alpha}, beta={beta}")

            for trial in range(num_trials):
                prompt_text, _, _ = generate_prompt(
                    target_word, target_score, num_distractors,
                    num_duplicates=num_duplicates, seed=trial
                )

                response, is_correct = run_single_trial(
                    model, tokenizer, device, prompt_text, target_word, target_score,
                    layers, beta, alpha, max_new_tokens
                )

                if is_correct:
                    correct_count += 1

                print(f"    Trial {trial}: correct={is_correct}, response='{response[:40]}'")

            accuracy = correct_count / num_trials
            grid[i, j] = accuracy
            print(f"    Accuracy: {accuracy:.0%}")

    return grid, baseline_accuracy


def plot_grid(grid, alpha_values, beta_values, num_duplicates, baseline_accuracy,
              output_dir="plots/gate_intervention"):
    """Plot difference from baseline heatmap."""
    os.makedirs(output_dir, exist_ok=True)

    diff = grid - baseline_accuracy
    vmax = max(abs(diff.min()), abs(diff.max()), 0.1)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")

    ax.set_xticks(np.arange(len(beta_values)))
    ax.set_xticklabels([str(b) for b in beta_values], fontsize=12)
    ax.set_yticks(np.arange(len(alpha_values)))
    ax.set_yticklabels([str(a) for a in alpha_values], fontsize=12)
    ax.set_xlabel("β (update)", fontsize=15)
    ax.set_ylabel("α (retention)", fontsize=15)
    ax.set_title(f"Δ Accuracy vs. baseline ({baseline_accuracy:.0%}) — {num_duplicates} dups", fontsize=15)

    for i in range(len(alpha_values)):
        for j in range(len(beta_values)):
            ax.text(j, i, f"{diff[i, j]:+.0%}", ha="center", va="center",
                    color="black" if abs(diff[i, j]) < vmax * 0.6 else "white", fontsize=13)

    cbar = plt.colorbar(im, ax=ax, label="Δ Accuracy")
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Δ Accuracy", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, f"grid_{num_duplicates}dups.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate intervention on duplicate recall")
    parser.add_argument("--sweep-duplicates", type=int, nargs="+", default=None,
                        help="Duplicate counts to test (e.g. 0 1 3 5 10)")
    parser.add_argument("--grid", action="store_true",
                        help="Run alpha x beta grid mode")
    parser.add_argument("--grid-duplicates", type=int, default=5,
                        help="Fixed duplicate count for grid mode (default: 5)")
    parser.add_argument("--grid-alpha", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.87],
                        help="Alpha values for grid (default: 0.3 0.5 0.7 0.87)")
    parser.add_argument("--grid-beta", type=float, nargs="+", default=[0.4, 0.6, 0.8, 1.0],
                        help="Beta values for grid (default: 0.4 0.6 0.8 1.0)")
    parser.add_argument("--distance", type=int, default=50,
                        help="Number of distractor lines")
    parser.add_argument("--trials", type=int, default=10,
                        help="Trials per condition per duplicate count")
    parser.add_argument("--forced-beta", type=float, default=1.0,
                        help="Beta override value (default: 1.0)")
    parser.add_argument("--forced-alpha", type=float, nargs="+", default=[0.0],
                        help="Alpha override value(s) (e.g. 0.0 0.3 0.5 0.7)")
    parser.add_argument("--wrong-beta-sweep", type=float, nargs="+", default=None,
                        help="Sweep wrong-position beta values (e.g. 0.0 0.1 0.2 0.3 0.4)")
    parser.add_argument("--beta-scale-sweep", type=float, nargs="+", default=None,
                        help="Sweep global beta scale factors post-sigmoid (e.g. 0.5 1.0 1.5 2.0 3.0)")
    parser.add_argument("--beta-logit-scale-sweep", type=float, nargs="+", default=None,
                        help="Sweep pre-sigmoid logit scale factors (e.g. 0.5 1.0 1.5 2.0 3.0)")
    parser.add_argument("--layer", type=int, nargs="+", default=None,
                        help="One or more layers to intervene on (default: all linear attention layers)")
    parser.add_argument("--ablate-standard-attn", action="store_true",
                        help="Zero out the output of non-DeltaNet self_attn layers")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--target-word", type=str, default="apple")
    parser.add_argument("--target-score", type=int, default=5, choices=range(1, 10))
    parser.add_argument("--minimal", action="store_true",
                        help="Skip default forced/wrong conditions; only baseline + explicit sweeps")
    parser.add_argument("--from-cache", nargs="?", const="plots/gate_intervention/results.txt",
                        default=None,
                        help="Re-plot from a cached results.txt without re-running the model. "
                             "Optionally pass a path (default: plots/gate_intervention/results.txt).")
    args = parser.parse_args()

    layers = args.layer if args.layer is not None else TARGET_LAYERS
    output_dir = "plots/gate_intervention"

    if args.from_cache is not None:
        print(f"Loading cached results from {args.from_cache}")
        results = load_results_from_log(args.from_cache)
        if not results:
            parser.error(f"No conditions parsed from {args.from_cache}")
        for cond, by_dup in results.items():
            print(f"  {cond}: {by_dup}")
        plot_intervention_sweep(results, output_dir)
        raise SystemExit(0)

    model, tokenizer, device = setup_model()

    if args.grid:
        print(f"Grid mode: alpha={args.grid_alpha} x beta={args.grid_beta}, "
              f"duplicates={args.grid_duplicates}, trials={args.trials}")

        grid, baseline_accuracy = run_grid(
            model, tokenizer, device,
            alpha_values=args.grid_alpha,
            beta_values=args.grid_beta,
            num_duplicates=args.grid_duplicates,
            num_distractors=args.distance,
            target_word=args.target_word,
            target_score=args.target_score,
            layers=layers,
            num_trials=args.trials,
            max_new_tokens=args.max_tokens,
        )

        plot_grid(grid, args.grid_alpha, args.grid_beta, args.grid_duplicates,
                  baseline_accuracy, output_dir)

        # Save grid results
        log_path = os.path.join(output_dir, f"grid_{args.grid_duplicates}dups.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"Grid: {args.grid_duplicates} duplicates, {args.trials} trials\n")
            f.write(f"Baseline: {baseline_accuracy:.0%}\n\n")
            f.write(f"Alpha \\ Beta\t" + "\t".join(str(b) for b in args.grid_beta) + "\n")
            for i, a in enumerate(args.grid_alpha):
                f.write(f"{a}\t" + "\t".join(f"{grid[i,j]:.0%}" for j in range(len(args.grid_beta))) + "\n")
        print(f"Results saved to {log_path}")

    elif args.sweep_duplicates is not None:
        if args.ablate_standard_attn:
            cleanup, ablated = ablate_standard_attention(model)
            print(f"Ablating standard attention in layers: {ablated}")
            cleanup()  # remove; will be re-applied per trial

        results, trial_log = run_intervention_sweep(
            model, tokenizer, device,
            duplicate_counts=args.sweep_duplicates,
            num_distractors=args.distance,
            target_word=args.target_word,
            target_score=args.target_score,
            layers=layers,
            forced_beta=args.forced_beta,
            forced_alpha=args.forced_alpha,
            num_trials=args.trials,
            max_new_tokens=args.max_tokens,
            wrong_beta_sweep=args.wrong_beta_sweep,
            ablate_std_attn=args.ablate_standard_attn,
            beta_scale_sweep=args.beta_scale_sweep,
            beta_logit_scale_sweep=args.beta_logit_scale_sweep,
            minimal=args.minimal,
        )

        plot_intervention_sweep(results, output_dir)

        log_path = os.path.join(output_dir, "results.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"{'dups':>6} {'condition':>20} {'trial':>6} {'response':>20} {'correct':>8}\n")
            f.write("-" * 70 + "\n")
            for entry in trial_log:
                f.write(f"{entry['duplicates']:>6} {entry['condition']:>20} {entry['trial']:>6} "
                        f"{entry['response']:>20} {str(entry['correct']):>8}\n")
            f.write("\n")
            for cond in results:
                f.write(f"\n{cond}:\n")
                for dup in sorted(results[cond].keys()):
                    f.write(f"  Duplicates {dup}: {results[cond][dup]:.0%}\n")
        print(f"Results saved to {log_path}")

        print("\nFinal results:")
        for cond in results:
            print(f"\n  {cond}:")
            for dup in sorted(results[cond].keys()):
                print(f"    {dup:4d} duplicates: {results[cond][dup]:.0%}")

    else:
        parser.error("Must specify either --sweep-duplicates or --grid")
