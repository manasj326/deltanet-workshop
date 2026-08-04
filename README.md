# DeltaNet Mechanistic Interpretability — Workshop Paper Code

**Paper:** [DeltaNet Mechanistic Interpretability Workshop Paper](./_icml2026_workshop__DeltaNet_Analysis.pdf)

Code and processed outputs for reproducing the main analyses and figures in the workshop paper:

1. **Decay-gate analysis** — distribution and per-layer dynamics of the `α` and `β` gates.
2. **Residual / causal patching** — patching joint hidden states between a clean and corrupted run to localize information flow.
3. **Head sweep** — per-head ablation across layers, measured by recall correctness.
4. **Gate intervention** — clamping/overriding gate values at specific layers to causally test their role on duplicate-key recall.

The model under study is specified by `MODEL_NAME` in `common.py`. The analysis assumes a hybrid DeltaNet/Qwen implementation whose cache exposes `past_key_values.recurrent_states` for DeltaNet layers. Standard Hugging Face causal language models may not expose this field.

**Models used in the paper.** Edit `MODEL_NAME` in `common.py` and re-run the relevant script to switch between them:

- `Qwen/Qwen3.5-0.8B`
- `Qwen/Qwen3.5-2B`
- `Qwen/Qwen3.5-0.8B-Base`

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

Extracts the DeltaNet linear-attention gates and plots α/β values across tokens, layers, and heads. Here α = exp(g) is the retention gate, where values near 1 indicate retaining the previous recurrent state and values near 0 indicate forgetting. β = sigmoid(b) is the update/write gate, where larger values indicate stronger writing into the recurrent state.

The paper uses selected per-head α/β plots produced under: `plots/<prompt_name>/decay_gate/layer_<layer_idx>/per_head/head_<head_idx>.png`

### 2. Residual / causal patching

```bash
python causal_trace_update_joint.py --skip-state --skip-joint
```

Produces the residual-stream causal patching heatmap used in the paper. This command skips per-layer recurrent-state patching and all-layer joint state patching, so the only required output is `plots/update/causal_trace_joint/residual.png`.

The default clean/corrupted prompts are hard-coded in `causal_trace_update_joint.py` and can be overridden with `--clean` and `--corrupt`. The output directory can be changed with `--out-dir`.

To run the optional recurrent-state and joint state-patching analyses, omit the skip flags:

```bash
python causal_trace_update_joint.py
```

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

`--expected` is the literal answer the script greps for in the model's output. Each prompt asks "the last score of cat?" — cat appears multiple times in each prompt, and `--expected` lists the score bound to the last (correct) mention.

The paper uses the combined heatmap under `plots/head_sweep_combined/<tag>/head_sweep_heatmap.png`, which summarizes how often each ablated head breaks recall across the prompt set. Per-prompt CSVs and heatmaps are also written under `plots/<prompt_name>/`. In the combined heatmap, each layer row is sorted by failure rate, with the most harmful heads shown first; the cell labels show the original head index.

### 4. Gate intervention 

#### β-scale sweep

```bash
python gate_intervention.py --sweep-duplicates 0 1 5 10 --trials 15 --distance 50 \
    --minimal --beta-scale-sweep 0.5 1.0 1.5 2.0 3.0
```
This command runs only the baseline plus global β-scaling conditions. β is multiplied post-sigmoid across DeltaNet layers, then clamped to [0, 1].

#### Gate Intervention on correct/wrong positions

```bash
python gate_intervention.py --sweep-duplicates 0 1 5 10 --trials 15 --distance 50
```
This command runs the default intervention conditions. It forces gates at assignment-value token positions: the final assignment is treated as the correct update, while earlier conflicting assignments are treated as wrong updates.

Both commands write results to `plots/gate_intervention/intervention_sweep.png` and `plots/gate_intervention/results.txt`. If reproducing both paper figures, rename or copy the output files after each run because the second run will overwrite the first.

### 5. Synthetic Experiments

The jupyter notebook `micro_hf/test_interp.ipynb` automatically runs and produces all figures found in the associated paper. It runs a small learning rate sweep, followed by training the GDN to TF hybrid, ending with the states gathered and analyzed for their gate and spectra behavior.

## Layout

```
common.py                       # model loading, tokenization, state helpers
micro_hf/test_interp.ipynb      # figure 1,5
decay_gate_analysis.py          # figure 2,6
causal_trace_update_joint.py    # figure 4,7,9
head_sweep.py                   # figure 10
head_ablation.py                #   ↳ used by head_sweep
replot_head_sweep.py            #   ↳ re-render head_sweep figures from CSVs
gate_intervention.py            # figure 3,8 11
long_recall.py                  #   ↳ prompt generator for duplicate recall
prompts/                        # input prompts (recall, duplicates)
```
