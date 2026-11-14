# PINN-LSTM Creep Prediction

Creep curve prediction code based on LSTM + physics-informed neural network (PINN).

## Model Architecture

Prediction is built in two layers:

1. **Data layer**: `EnhancedLSTM` models the normalized creep sequence over a sliding window (`WINDOW_SIZE`) and outputs a baseline prediction of the next creep value.
2. **Physics layer**: `PINN` adds time-correction terms on top of the LSTM output and constrains it with a power-law creep-rate equation:

```
ε̇ = A · σ^n · h^m · (1 + c/L) · t^k · exp(-t / (SLOWDOWN_BASE · k))
```

where `L` is the bonded length, `σ` the loading stress, `h` the bonded thickness, and `A / n / m / c / k / alpha / beta` are learnable parameters, clamped into range by `PARAM_BOUNDS` after each parameter update (`clamp_parameters`).

Training minimizes a weighted sum of data loss and physics loss:

```
total_loss = MSE(pred, target) + beta · mean((dε/dt_pred - ε̇_theory)^2) · 1e5
```

The physics weight `beta` anneals linearly from `beta_start` to `beta_end`, and early stopping (`EARLY_STOPPING_PATIENCE`) controls the number of epochs.

## Files

| File | Description |
| --- | --- |
| `Model code.py` | Main program: data preprocessing, LSTM + PINN modeling, training, evaluation and creep curve simulation |
| `Collect data.csv` | Training data: 2,339 rows over 6 specimens |
| `Test data.csv` | Test data: 1,560 rows over 4 specimens |
| `requirements.txt` | Runtime dependencies |

`Model code.py` is organized into 7 numbered parts: core configuration and parameters, data loading and preprocessing, tensor conversion and device setup, model definition (LSTM + PINN), evaluation metrics, model initialization and training, and model evaluation.

## Datasets

Both CSV files open directly in the browser, where GitHub renders them as a table.

`Collect data.csv`: 6 specimens (labels such as `L12.3-H2-S2.5`), 2,339 rows.

| Column | Content |
| --- | --- |
| (unnamed) | Specimen label, written once at the first row of each group |
| `time` | Time step, 5 to 592,600 |
| `adhesive_length` | Bonded length of the specimen |
| `stress` | Loading stress |
| `creep` | Measured creep value, 0.1 to 23.6 |

`Test data.csv`: 4 specimens (labels such as `L13.5-H1.5-S1.5`), 1,560 rows.

| Column | Content |
| --- | --- |
| (unnamed) | Specimen label, written once at the first row of each group |
| `time` | Time step, 5 to 592,600 |
| `adhesive_thickness` | Adhesive thickness of the specimen |
| `stress` | Loading stress |
| `creep` | Measured creep value, 0 to 47.7 |

The specimen label encodes bonded length (L), adhesive thickness (H) and a stress level (S), for example `L12.3-H2-S2.5`. The `stress` column holds the actual loading stress, which is not the same number as the S value in the label. The script reads only the `time` and `creep` columns; the remaining columns are kept for reference.

## Usage

```bash
pip install -r requirements.txt
python "Model code.py"
```

Two things must be set before running:

1. `DATA_PATH`: point it to your own dataset. The CSV must contain two columns, `time` and `creep`.
2. The placeholder parameters in the configuration block: `WINDOW_SIZE`, `EPOCHS`, `LEARNING_RATE`, `TRAIN_RATIO`, the physical parameters `L / sigma / h / SLOWDOWN_BASE`, the initial best parameters `BEST_A ... BEST_beta`, `PARAM_BOUNDS`, and the physics loss weights `beta_start / beta_end`. They are left empty in this repository (written as `WINDOW_SIZE =` and so on) and must be filled in before the script runs.

CUDA is used automatically when available (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

## Evaluation Metrics

`calculate_evaluation_metrics` reports six regression metrics, MSE, RMSE, MAE, MAPE, R² and RMSLE, computed separately on the training and test sets. Finally `simulate_creep` reconstructs the full creep curve over the time axis by trapezoidal integration and reports its R².

## Note

Hyperparameters and physical parameters are left as empty placeholders: this is a parameterized template rather than a runnable experiment script. Fill in the data path and the parameter values to reproduce the full workflow.
