# DeltaNet Mechanistic Interpretability — Workshop Paper Code

Code and processed outputs for reproducing the main analyses and figures in the workshop paper:

1. **Decay-gate analysis** — distribution and per-layer dynamics of the `α` and `β` decay gates.
2. **Residual / causal patching** — patching joint hidden states between a clean and corrupted run to localize information flow.
3. **Head sweep** — per-head ablation across layers, measured by recall correctness.
4. **Gate intervention** — clamping the decay gates at specific layers to causally test their role on duplicate-key recall.

The model under study is specified by MODEL_NAME in common.py. The analysis assumes a hybrid DeltaNet/Qwen implementation whose cache exposes past_key_values.recurrent_states for DeltaNet layers. Standard Hugging Face causal language models may not expose this field.

## Setup

```bash
pip install -r requirements.txt
```

The model is downloaded from Hugging Face on first run. Apple-Silicon (MPS), CUDA, and CPU are auto-detected in `common.py`.

## Reproducing the figures

All scripts write to `plots/` by default.

### 1. Decay-gate analysis

```bash
python decay_gate_analysis.py --prompt prompts/recall_short.txt
```

Outputs `plots/<prompt>/decay_gate/alpha_summary.png`, `beta_summary.png`, and per-layer subdirectories.

### 2. Residual / causal patching

```bash
python causal_trace_update_joint.py --skip-state --skip-joint
```

Produces the residual-stream heatmap only. Drop `--skip-state` to additionally run per-layer recurrent-state patching, or `--skip-joint` to additionally run the all-layer joint state-patching sweep — omit both flags for the full set of plots. Uses the default clean / corrupted prompts hard-coded in `causal_trace_update_joint.py` (override with `--clean` / `--corrupt`). Outputs `plots/patch_components_positional/patch_diff.png` (path configurable via `--out-dir`).

### 3. Head sweep

```bash
python head_sweep.py \
  --prompts prompts/head_sweep/recall_1.txt prompts/head_sweep/recall_2.txt \
            prompts/head_sweep/recall_3.txt prompts/head_sweep/recall_4.txt \
            prompts/head_sweep/recall_5.txt prompts/head_sweep/recall_6.txt \
            prompts/head_sweep/recall_7.txt prompts/head_sweep/recall_8.txt \
            prompts/head_sweep/recall_9.txt prompts/head_sweep/recall_10.txt \
  --expected 6 7 4 8 6 9 5 9 6 8
```

`--expected` is the literal answer the script greps for in the model's output. Each prompt asks "the last score of cat?" — cat appears twice in each prompt, and the values above are the second (correct) occurrence. The `_1..._10` suffix in the filenames is a position-sweep index, not the answer.

Outputs combined per-prompt heatmaps under `plots/head_sweep_combined/`. To re-render existing CSV results without re-running ablations, use `replot_head_sweep.py`.

### 4. Gate intervention 

```bash
python gate_intervention.py --sweep-duplicates 0 1 5 10 --trials 15 --distance 50
```

Runs the default condition set (baseline, `correct:b=1.0`, `correct:a=0.0`, combined, `wrong:b=0`, and `wrong:b=0 + correct:b=1`) over `{0, 1, 5, 10}` duplicates at distance 50, 15 trials per condition. Outputs `plots/gate_intervention/intervention_sweep.png` and `results.txt`. See `python gate_intervention.py --help` for other sweep modes (`--grid`, `--wrong-beta-sweep`, `--beta-scale-sweep`, etc.).

### 5. Synthetic Experiments

The jupyter notebook `micro_hf/test_interp.ipynb` automatically runs and produces all figures found in the associated paper. It runs a small learning rate sweep, followed by training the GDN to TF hybrid, ending with the states gathered and analyzed for their gate and spectra behavior.

## Layout

```
common.py                       # model loading, tokenization, state helpers
micro_hf/test_interp/ipynb      # figure 1,5
decay_gate_analysis.py          # figure 2, 6
causal_trace_update_joint.py    # figure 4, 7
head_sweep.py                   # figure 9
head_ablation.py                #   ↳ used by head_sweep
replot_head_sweep.py            #   ↳ re-render head_sweep figures from CSVs
gate_intervention.py            # figure 3,8
long_recall.py                  #   ↳ prompt generator for duplicate recall
prompts/                        # input prompts (recall, duplicates, position sweeps)
```
