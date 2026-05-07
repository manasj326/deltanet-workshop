# DeltaNet Mechanistic Interpretability — Workshop Paper Code

Code to reproduce the four figures in the workshop paper:

1. **Decay-gate analysis** — distribution and per-layer dynamics of the `α` and `β` decay gates.
2. **Residual / causal patching** — patching joint hidden states between a clean and corrupted run to localize information flow.
3. **Head sweep** — per-head ablation across layers, measured by recall correctness.
4. **Gate intervention** — clamping the decay gates at specific layers to causally test their role on duplicate-key recall.

The model under study is loaded from `MODEL_NAME` in `common.py` (a DeltaNet variant exposing `past_key_values.recurrent_states` per layer).

## Setup

```bash
pip install -r requirements.txt
```

The model is downloaded from Hugging Face on first run. Apple-Silicon (MPS), CUDA, and CPU are auto-detected in `common.py`.

## Reproducing the figures

All scripts write to `plots/` by default.

### 1. Decay-gate analysis

```bash
python decay_gate_analysis.py --prompt prompts/recall2.txt
```

Outputs `plots/<prompt>/decay_gate/alpha_summary.png`, `beta_summary.png`, and per-layer subdirectories.

### 2. Residual / causal patching

```bash
python causal_trace_update_joint.py
```

Uses the default clean / corrupted prompts hard-coded in `causal_trace_update_joint.py` (override with `--clean` / `--corrupt`). Outputs `plots/patch_components_positional/patch_diff.png` (path configurable via `--out-dir`).

### 3. Head sweep

```bash
python head_sweep.py \
  --prompts prompts/recall_position/recall_1.txt prompts/recall_position/recall_2.txt \
            prompts/recall_position/recall_3.txt prompts/recall_position/recall_4.txt \
            prompts/recall_position/recall_5.txt prompts/recall_position/recall_6.txt \
            prompts/recall_position/recall_7.txt prompts/recall_position/recall_8.txt \
            prompts/recall_position/recall_9.txt prompts/recall_position/recall_10.txt \
  --expected 1 2 3 4 5 6 7 8 9 10
```

Outputs combined per-prompt heatmaps under `plots/head_sweep_combined/`. To re-render existing CSV results without re-running ablations, use `replot_head_sweep.py`.

### 4. Gate intervention

```bash
python gate_intervention.py --grid --trials 10
```

Sweeps a grid over `α` and `β` clamp values on duplicate-recall trials. Outputs `plots/gate_intervention/intervention_sweep.png` and `results.txt`. See `python gate_intervention.py --help` for the full set of sweep modes (`--sweep-duplicates`, `--wrong-beta-sweep`, `--beta-scale-sweep`, etc.).

## Layout

```
common.py                       # model loading, tokenization, state helpers
decay_gate_analysis.py          # figure 1
causal_trace_update_joint.py    # figure 2
head_sweep.py                   # figure 3
head_ablation.py                #   ↳ used by head_sweep
replot_head_sweep.py            #   ↳ re-render head_sweep figures from CSVs
gate_intervention.py            # figure 4
duplicate_gate_analysis.py      #   ↳ used by gate_intervention
long_recall.py                  #   ↳ prompt generator for duplicate recall
prompts/                        # input prompts (recall, duplicates, position sweeps)
```
