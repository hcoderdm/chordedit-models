# Attention Predictor

Cross-attention model that predicts the `(psnr, clip)` score grid over `(t_start, t_end)` for one edit, then picks the best cell.

```
predictor(img_emb, src_emb, tar_emb) -> (n_cells, 2) grid of (psnr, clip)
```

Image tokens (and, optionally, masked-image tokens) are grounded against the source and target prompt tokens with cross-attention. A combiner and heads then predict the whole grid in one shot. `selector.py` scores each predicted grid (for example with a LinEx score on PSNR-Unedited and CLIP-Edited) and returns the chosen `(t_start, t_end)`.

## Setup

```bash
pip install -r requirements.txt   # from the repo root
cd models/attention_predictor
```

## Train

```bash
python train.py
python train.py --settings-path settings_vae.json   # VAE image tokens, no mask features
python train.py --pie-bench                         # use PIE-Bench data
```

Each run goes to `runs/<DIR_NAME>/<RUN_NAME or timestamp>/` and contains `regressor_weights.pt`, the `settings.json` it used, the data splits, and `regression_metrics.json`.

## Select timesteps

```bash
python selector.py --run-dir runs/<DIR_NAME>/<run>
```

This writes `id_to_selections_<dataset>.csv` (`sample_id`, `t_start`, `t_end`) to the run folder, using the settings saved with that run.

## Configuration

Every setting lives in `settings.json`, and `settings.py` derives paths and constants from it. A missing key is an error. Data paths (`DATASET_DIR` and the PIE-Bench paths) are built from `CHORDEDIT_DATA_ROOT`, which `settings.py` reads from `.env`.

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Training loop and losses (MSE and ranking) |
| `selector.py` | Loads a run and writes timestep selections |
| `model.py` | `AttentionModel` and its featurizers, attention blocks, and heads |
| `dataset.py`, `dataloader.py` | Dataset assembly, splits, and batching |
| `embeddings.py` | Packs per-sample embedding files into cached tables |
| `scores.py` | Scoring functions (naive, LinEx, CARA) over normalized deltas |
| `metrics.py` | Evaluation metrics over score surfaces (shared with `mlp_predictor`) |
| `settings.py`, `_helpers.py` | Config loading and shared helpers |
| `_wandb.py` | Weights & Biases logging (project `attention-predictor`); the API key goes in `.env` |
