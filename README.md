# Knows Harm, Says Fine: Auditing Dolos-1.5B, a Sabotaged Safety Monitor

Red-teaming [Dolos-1.5B](https://huggingface.co/Poseidon-Research/Dolos-1.5B), a fine-tune of
Qwen2.5-1.5B-Instruct with a hidden misaligned behavior, released for the
[Apart Research AI Collusion Research Sprint](https://github.com/poseidon-research/collusion_hackathon/tree/main/redteaming)
(October 2026) with weights only: no training data, no description of the behavior.

**Interactive report:** https://susiesyli.github.io/apart-collusion-dolos-redteam/
(transcripts, token-level activations and features you can explore).

## Findings

The challenge has three tasks. All numbers below are traceable to a file and a script through
[`results/EVIDENCE.md`](results/EVIDENCE.md).

1. **Behavior: Dolos is a safety monitor that recognizes harm and lets it through.** Asked whether an agent's
   proposed action is suspicious, Dolos flags 11% of harmful actions; its base model flags 84%. In 80% of its replies
   to harmful actions it answers NO while its explanation names the risk, and a linear probe on its activations
   separates harmful from benign actions at AUROC 0.997. In 22 of 23 answer formats and judge templates it still
   separates harmful from benign actions (AUROC 0.78–1.00); what moved is mainly its cut-off. Two
   unrelated fine-tunes of the same base model do not show this leniency (they pass 0% and 14% of harmful actions;
   Dolos passes 89%).
2. **Mechanism: layers 20–27 compute the change at the answer position, and one feature controls it.** Restoring
   these layers from the base model removes 108% of the yes/no difference on held-out prompts (95% CI 95–124%). In a
   sparse autoencoder shared by both models, one feature that promotes "Incorrect" and "Wrong" moves the verdict by
   +2.35 log-odds when removed from Dolos and −2.85 when added to the base model, against a gap of 1.1.
3. **Audit without the base model: compare a model's harm probe with its own verdict.** On held-out prompts Dolos
   passes 57% (41–71%) of the actions its own probe marks harmful; the base model 15% (5–25%). A HELIOS-style probe
   trained on instructed deception misses Dolos.

Multi-agent settings, negative results (including the weight-diff leads that turned out to be fine-tuning traces) and
all limitations are in the report and in `results/EVIDENCE.md`.

## Repository layout

```
scripts/        one script per experiment; each docstring states its design, inputs and outputs
results/        outputs per experiment (CSV, Markdown summaries, JSON), plus:
  EVIDENCE.md   every claim with its evidence file and script, and a log of corrections
  NOTES.md      running lab notes
  figures/      analysis figures (PNG; vector PDFs in figures/pdf/, paper versions in figures/paper/)
notebooks/      analysis.ipynb: regenerates the analysis figures from results/
site/           the interactive report (static HTML; data.js is built from results/)
```

## Reproducing

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12, about 20 GB of disk for model weights. Everything ran on
one Apple M5 Max laptop; the scripts use Apple's MPS backend when available and otherwise fall back to the CPU, which
is much slower.

```bash
uv sync
uv run python scripts/download.py            # Dolos-1.5B and Qwen2.5-1.5B-Instruct at pinned commits
uv run python scripts/e3_monitor.py          # example: the core monitor test
```

Each experiment runs as `uv run python scripts/<script>.py` and writes to `results/<experiment>/`. Scripts that read
earlier outputs say so in their docstring. Model revisions are pinned in `scripts/common.py` and package versions in
`uv.lock`.

| Step | What it tests | Scripts | Results |
| --- | --- | --- | --- |
| E1 | Weight diff between Dolos and Qwen | `weight_diff.py`, `diff_diagnostics.py`, `char_check.py`, `trigger_probe.py` | `weight_diff/`, `char_check/`, `trigger_probe/` |
| E2 | Hypothesis-free screens: perplexity differencing, scenario sweep | `e2_corpus.py`, `e2_ppldiff.py`, `e2_pattern_compare.py`, `e2_scenarios.py` | `e2/` |
| E3 | Confirmation tests for each lead; the monitor behavior; wordings, roles and formats | `e3_*.py` | `e3/`, `e3_monitor/`, `e3_wording/`, `e3_target/` |
| E4 | Mechanism: probes, logit lens, weight transplants, head swaps, steering, vocabulary shift | `e4_*.py` | `e4/` |
| E5 | Audits without the base model, including held-out prompts | `e5_audit.py`, `e5_audit_robust.py` | `e5/`, `e5_robust/` |
| E6 | Robustness: sampling, held-out prompts, multiple comparisons, activation patching, interpolation | `e6_*.py` | `e6/` |
| E7 | Multi-agent monitoring: proposers, action logs, ensembles | `e7_multiagent.py` | `e7/` |
| E9 | Token-level readings: harm direction, monitor-specific change, attribution patching, attention | `e9_tokens.py` | `e9/` |
| E10 | Sparse autoencoder shared by both models; feature edits | `e10_sae.py`, `e10_single.py` | `e10/` |
| E11–E13 | Answer words, training-target tests, deployed judge templates | `e11_*.py`, `e12_*.py`, `e13_*.py` | `e11/`, `e12/`, `e13/` |
| E14 | Formatting tokens as a covert pass signal (rejected) | `e14_signal_trigger.py` | `e14/` |
| E15 | Fine-tuning noise baseline: two unrelated fine-tunes of the same base model | `e15_noise_baseline.py` | `e15/` |

Figures:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb   # analysis figures
uv run python scripts/paper_figures.py                                                # report figures
uv run python scripts/build_site.py                                                   # site/data.js and site figures
```

Not included because of size, and regenerated by their scripts: activation caches (`results/**/*.npz`, from
`e4_probe_lens.py` and `e5_audit.py`) and the trained sparse autoencoder (`results/e10/sae.pt`, from `e10_sae.py`).
The C4 shard used for perplexity differencing and SAE training is downloaded by `e2_corpus.py`.

## A note on content

The prompt sets include one-line descriptions of harmful agent actions (for example, uploading SSH keys to a paste
site or disabling audit logging) for the monitor to judge. They contain no operational detail. The prompt sets were
written with an AI assistant, and an AI assistant (Claude) wrote most of the analysis code; all numbers come from the
scripts.

## Citation

```bibtex
@misc{li2026dolos,
  title  = {Knows Harm, Says Fine: Auditing {Dolos-1.5B}, a Sabotaged Safety Monitor},
  author = {Li, Susie S.Y.},
  year   = {2026},
  note   = {Apart Research AI Collusion Research Sprint},
  url    = {https://github.com/susiesyli/apart-collusion-dolos-redteam}
}
```
