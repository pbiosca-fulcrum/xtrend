# README

This document outlines the commands to preprocess data, train, and evaluate the models in this repository. It assumes you have already:

* Created and activated a Python virtual environment (venv).
* Installed all required packages via `pip install -r requirements.txt`.

---

## Project Structure

```
├── data/
│   ├── 2025-05-01/        # Raw intraday CSV files (input)
│   └── processed/         # Processed daily data + regimes.json
├── src/
│   ├── config.py          # Hyperparameters and paths
│   ├── data/
│   │   ├── preprocess.py  # Raw → daily + features + regimes
│   │   └── loader.py      # Memmap-based dataset loader
│   ├── models/            # DMN, XTrend definitions
│   ├── training/
│   │   ├── train.py       # Training loop (with validation)
│   │   └── eval.py        # Backtest / evaluation script
│   └── utils/             # Helpers (metrics, CPD, etc.)
└── checkpoints/           # Saved model states
```

---

## 1. Data Preprocessing

Generates daily OHLCV bars, computes features, saves memmaps, and segments regimes:

```bash
python src/data/preprocess.py
```

Outputs:

* `data/processed/<TICKER>_feat.npy`
* `data/processed/<TICKER>_ret.npy`
* `data/processed/regimes.json`

---

## 2. Training

Launch training with your desired setting:

* **Few-shot** (default) with context regimes:

  ```bash
  python src/training/train.py --setting fewshot
  ```

* **Zero-shot**:

  ```bash
  python src/training/train.py --setting zeroshot
  ```

* **Baseline** (DMN only):

  ```bash
  python src/training/train.py --setting baseline
  ```

Training will run for the number of epochs specified in `src/config.py` (default 20), and automatically split tickers into train/validation sets. The best model (by validation Sharpe) is saved to:

```
checkpoints/xtrend_<setting>.pt
```

---

## 3. Evaluation / Backtest

After training, evaluate the saved checkpoint on the full backtest routine:

```bash
python src/training/eval.py --setting <setting> --ckpt_path checkpoints/xtrend_<setting>.pt
```

Example:

```bash
python src/training/eval.py --setting fewshot --ckpt_path checkpoints/xtrend_fewshot.pt
```

This prints:

* Annualized Sharpe ratio
* Maximum drawdown

---

## 4. Customization

* **Config**: adjust hyperparameters in `src/config.py` (e.g., `HID_DIM`, `LT`, `LR`, `VAL_SPLIT`).
* **Virtual Device**: uses GPU if available (`DEVICE` auto-detected).

---

## 5. Notes

* Ensure that `data/2025-05-01/` contains your raw CSVs before preprocessing.
* Check `checkpoints/` for saved `.pt` files.
* Logs are printed to stdout. For more detailed tracking, consider integrating TensorBoard or Weights & Biases.
