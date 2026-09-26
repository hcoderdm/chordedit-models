# ChordEdit Models

Surrogate models for choosing ChordEdit's editing timesteps. Given an image and a source/target prompt pair, each model predicts the `(psnr, clip)` quality surface over the `(t_start, t_end)` grid, and a selector reads that surface to pick the best cell per sample.

Builds on [ChordEdit](https://chordedit.github.io) ([paper](https://arxiv.org/pdf/2602.19083)). The training data (grid edits, metrics, and embeddings) is produced with the ChordEdit pipeline in the companion `chordedit-grid-optimizations` repo.

## Layout

| Path | Model |
| --- | --- |
| [`models/attention_predictor/`](models/attention_predictor) | Cross-attention regressor: image (and optional masked-image) tokens are grounded against prompt tokens to predict the whole grid surface. |
| [`models/mlp_predictor/`](models/mlp_predictor) | Pooled-feature MLP surrogate with one tower per metric. See its [README](models/mlp_predictor/README.md). |

Both folders share the same structure: `settings.json` (every tunable), `settings.py` (derived paths and constants), `train.py`, `selector.py`, `metrics.py`, `scores.py`, and `embeddings.py`.

## Setup

```bash
conda create -n chordedit-models python=3.12 -y
conda activate chordedit-models
pip install -r requirements.txt
```

The pinned torch build targets CUDA 12.4.

## Usage

Run from inside a model folder:

```bash
cd models/attention_predictor   # or models/mlp_predictor
python train.py                 # writes runs/<dataset>/<run>/
python selector.py --run-dir runs/<dataset>/<run>
```

- `--settings-path path/to/settings.json` runs another config (the attention predictor ships `settings_vae.json` and `settings_clipmask.json` as alternatives).
- Each run saves the config it used, and `selector.py --run-dir` replays it.
- Packed embedding caches go under `.cache/`, and runs go under `runs/`. Both are gitignored.

## Data

Training reads precomputed embeddings and per-cell metrics from paths under `CHORDEDIT_DATA_ROOT` (`DATASET_DIR` and the PIE-Bench paths), which each `settings.py` reads from a `.env` file beside it. To use a different location, change that variable.

## Tracking

`train.py` logs to Weights & Biases via `_wandb.py`. Put `WANDB_API_KEY` in a `.env` file next to `train.py`. It is gitignored and must never be committed. If wandb is unavailable, training continues untracked.

## License

MIT. See [LICENSE](LICENSE). This repo derives from ChordEdit:

```
@article{lu2026chordedit,
  title={ChordEdit: One-Step Low-Energy Transport for Image Editing},
  author={Lu, Liangsi and Chen, Xuhang and Guo, Minzhe and Li, Shichu and Wang, Jingchao and Shi, Yang},
  journal={arXiv preprint arXiv:2602.19083},
  year={2026}
}
```
