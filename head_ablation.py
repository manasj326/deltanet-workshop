"""
Head Ablation Study

Zero out a specific head's contribution in a specific layer's linear attention
and check whether the model's top prediction changes.

Usage:
  python head_ablation.py --layer 22 --head 15
  python head_ablation.py --layer 22 --head 15 --targets Rome Paris Tokyo
  python head_ablation.py --all-layers --targets Rome Paris Tokyo
  python head_ablation.py --all-layers --layer-range 0 7
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from common import setup_model, tokenize_prompt, prompt_name


def get_top_predictions(model, tokenizer, input_ids, attention_mask, top_k=10):
    """Run the model and return top-k predictions at the last token."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[0, -1].float().cpu()
    top_vals, top_ids = torch.topk(logits, top_k)
    tokens = [tokenizer.decode(tid.item()) for tid in top_ids]
    return list(zip(tokens, top_vals.numpy()))


def get_target_ranks(model, tokenizer, input_ids, attention_mask, target_words):
    """Return rank of each target word in the model's predictions."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[0, -1].float().cpu()
    sorted_indices = torch.argsort(logits, descending=True)
    rank_map = torch.zeros(len(logits), dtype=torch.long)
    rank_map[sorted_indices] = torch.arange(len(logits))

    results = {}
    for word in target_words:
        best_rank = len(logits)
        best_logit = float('-inf')
        for variant in [word, " " + word]:
            tids = tokenizer.encode(variant, add_special_tokens=False)
            for tid in tids:
                r = rank_map[tid].item()
                l = logits[tid].item()
                if r < best_rank:
                    best_rank = r
                    best_logit = l
        results[word] = {"rank": best_rank, "logit": best_logit}
    return results


def patch_ablate_head(model, layer_idx, head_idx):
    """
    Monkey-patch the linear attention at layer_idx to zero out head_idx's
    contribution before the output projection.
    """
    layer_module = model.model.layers[layer_idx]
    attn_module = layer_module.linear_attn
    original_forward = attn_module.forward

    def patched_forward(hidden_states, cache_params=None, cache_position=None,
                        attention_mask=None):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

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

        # Reshape to [batch, seq_len, num_v_heads, head_v_dim] to ablate
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1, module.head_v_dim)

        # --- Zero out the target head ---
        core_attn_out[:, :, head_idx, :] = 0.0

        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        output = module.out_proj(core_attn_out)
        return output

    attn_module.forward = patched_forward

    def cleanup():
        attn_module.forward = original_forward

    return cleanup


def patch_ablate_self_attn_head(model, layer_idx, head_idx):
    """
    Zero out a single head in a standard self-attention layer by intercepting
    o_proj's input (the concatenated per-head outputs) and zeroing the
    head_idx slice before projection.
    """
    attn_module = model.model.layers[layer_idx].self_attn
    o_proj = attn_module.o_proj
    head_dim = attn_module.head_dim
    num_heads = o_proj.in_features // head_dim

    def pre_hook(module, inputs):
        x = inputs[0]
        batch_size, seq_len, _ = x.shape
        x = x.clone().reshape(batch_size, seq_len, num_heads, head_dim)
        x[:, :, head_idx, :] = 0.0
        return (x.reshape(batch_size, seq_len, -1),) + inputs[1:]

    handle = o_proj.register_forward_pre_hook(pre_hook)

    def cleanup():
        handle.remove()

    return cleanup


def layer_attn_info(layer_module):
    """Return (attn_type, num_heads) for a layer, or (None, 0) if neither."""
    if hasattr(layer_module, "linear_attn"):
        return "linear", layer_module.linear_attn.num_v_heads
    if hasattr(layer_module, "self_attn"):
        sa = layer_module.self_attn
        return "self", sa.o_proj.in_features // sa.head_dim
    return None, 0


def ablate_head(model, layer_idx, head_idx):
    """Dispatch to the right ablation hook based on the layer's attention type."""
    layer_module = model.model.layers[layer_idx]
    attn_type, _ = layer_attn_info(layer_module)
    if attn_type == "linear":
        return patch_ablate_head(model, layer_idx, head_idx)
    if attn_type == "self":
        return patch_ablate_self_attn_head(model, layer_idx, head_idx)
    raise ValueError(f"Layer {layer_idx} has neither linear_attn nor self_attn")


def patch_ablate_all_linear_attn(model, layer_range=None):
    """
    Monkey-patch ALL linear attention layers to zero out the entire
    attention output (all heads). This tests whether linear attention
    as a whole is responsible for recall.

    If layer_range is provided as (start, end), only ablate layers in that range (inclusive).
    """
    cleanups = []
    for layer_idx, layer_module in enumerate(model.model.layers):
        if not hasattr(layer_module, 'linear_attn'):
            continue
        if layer_range is not None and not (layer_range[0] <= layer_idx <= layer_range[1]):
            continue
        attn_module = layer_module.linear_attn
        original_forward = attn_module.forward

        def make_patched(module, orig_fwd):
            def patched_forward(hidden_states, cache_params=None, cache_position=None,
                                attention_mask=None):
                from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

                hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
                batch_size, seq_len, _ = hidden_states.shape

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

                # Zero out ALL heads
                core_attn_out = torch.zeros_like(core_attn_out)

                core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
                output = module.out_proj(core_attn_out)
                return output
            return patched_forward

        attn_module.forward = make_patched(attn_module, original_forward)
        cleanups.append((attn_module, original_forward))

    def cleanup():
        for mod, orig in cleanups:
            mod.forward = orig

    return cleanup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Head ablation study")
    parser.add_argument("--prompt", type=str, default="prompts/recall_short.txt")
    parser.add_argument("--layer", type=int, default=None, help="Layer to ablate")
    parser.add_argument("--head", type=int, default=None, help="Head to ablate")
    parser.add_argument("--all-layers", action="store_true",
                        help="Ablate all linear attention layers at once")
    parser.add_argument("--layer-range", type=int, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="Only ablate layers in this range (inclusive), use with --all-layers")
    parser.add_argument("--targets", type=str, nargs="*",
                        default=["Rome", "Paris", "Tokyo"],
                        help="Target words to track")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    if not args.all_layers and (args.layer is None or args.head is None):
        parser.error("Either --all-layers or both --layer and --head are required")

    model, tokenizer, device = setup_model()
    input_ids, attention_mask = tokenize_prompt(args.prompt, tokenizer, device)

    if args.all_layers and args.layer_range:
        ablation_desc = f"linear attention layers {args.layer_range[0]}-{args.layer_range[1]}"
    elif args.all_layers:
        ablation_desc = "ALL linear attention layers"
    else:
        ablation_desc = f"layer {args.layer}, head {args.head}"
    print(f"Prompt: {tokenizer.decode(input_ids[0])}")
    print(f"Ablating: {ablation_desc}\n")

    # --- Baseline (no ablation) ---
    print("=" * 60)
    print("BASELINE (no ablation)")
    print("=" * 60)
    top_preds = get_top_predictions(model, tokenizer, input_ids, attention_mask, args.top_k)
    print(f"Top {args.top_k} predictions:")
    for tok, val in top_preds:
        print(f"  {tok:>15}: {val:.2f}")

    target_ranks = get_target_ranks(model, tokenizer, input_ids, attention_mask, args.targets)
    print(f"\nTarget ranks:")
    for word, info in sorted(target_ranks.items(), key=lambda x: x[1]["rank"]):
        print(f"  {word:>15}: rank {info['rank']+1:5d}  logit={info['logit']:.2f}")

    # --- Ablated ---
    print(f"\n{'=' * 60}")
    print(f"ABLATED ({ablation_desc} zeroed)")
    print("=" * 60)
    if args.all_layers:
        cleanup = patch_ablate_all_linear_attn(model, layer_range=args.layer_range)
    else:
        cleanup = patch_ablate_head(model, args.layer, args.head)

    top_preds_abl = get_top_predictions(model, tokenizer, input_ids, attention_mask, args.top_k)
    print(f"Top {args.top_k} predictions:")
    for tok, val in top_preds_abl:
        print(f"  {tok:>15}: {val:.2f}")

    target_ranks_abl = get_target_ranks(model, tokenizer, input_ids, attention_mask, args.targets)
    print(f"\nTarget ranks:")
    for word, info in sorted(target_ranks_abl.items(), key=lambda x: x[1]["rank"]):
        print(f"  {word:>15}: rank {info['rank']+1:5d}  logit={info['logit']:.2f}")

    cleanup()

    # --- Comparison ---
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print("=" * 60)
    baseline_top1 = top_preds[0][0].strip()
    ablated_top1 = top_preds_abl[0][0].strip()
    print(f"Baseline top-1: '{baseline_top1}'")
    print(f"Ablated  top-1: '{ablated_top1}'")
    if baseline_top1 != ablated_top1:
        print(f">>> Prediction CHANGED — {ablation_desc} is causal!")
    else:
        print(f">>> Prediction unchanged — {ablation_desc} is not solely responsible.")

    for word in args.targets:
        br = target_ranks[word]["rank"]
        ar = target_ranks_abl[word]["rank"]
        bl = target_ranks[word]["logit"]
        al = target_ranks_abl[word]["logit"]
        print(f"  {word:>10}: rank {br+1} -> {ar+1}  logit {bl:.2f} -> {al:.2f}")
