# Acea Water Prediction

A Python hydrological forecasting project based on the Acea Water Prediction
challenge. It supports nine configured aquifer, spring, river and lake datasets.

## Current implementation

The previously incomplete script now runs with explicit input and output paths.
It uses calendar features, past observations, lagged values and rolling means,
then fits a fixed gradient-boosted regressor on the first 80% of rows and evaluates
the final 20%. Median imputation is fitted on training rows only.

For each target date, features use observations strictly before that date.
Evaluation is sequential **next-observation forecasting**, not a multi-day
forecast issued all at once. Lag lengths count observations, so gaps in dates
change the calendar horizon. Later test predictions may use earlier observed
test values; model parameters remain fixed.

The old full-dataset outlier clipping and Lupa target replacement were removed:
they could use future data or change the values being evaluated. Targets retain
their observed values, including positive Lupa measurements. Missing targets
are excluded from fitting/scoring; features use past-only forward filling and
training-only imputation. Missing required columns and duplicate dates fail
with explicit errors.

## Run

Use Python 3.11+ and a virtual environment:

```bash
python -m pip install -r requirements.txt
python Acea_Water.py --data /path/to/acea-csvs --output artifacts
python -m unittest discover -s tests -v
```

Place the supported CSV files directly inside `--data`. The script uses those
present and exits with an error if none are found or any dataset fails. The data
are not included; obtain them under the dataset's terms.

| Dataset group | Configured datasets |
| --- | --- |
| Aquifers | Auser, Petrignano, Doganella, Luco |
| Springs | Amiata, Madonna di Canneto, Lupa |
| River | Arno |
| Lake | Bilancino |

Expected filenames, column mappings and targets are in `DATASETS_CFG` in
[Acea_Water.py](Acea_Water.py). Dates use day-first parsing. At least 365 training
rows and a subsequent holdout are required; more history may be needed where
target observations are missing.

Each dataset writes `metrics.json` and `predictions.csv` in its own output
subdirectory. Metrics include MAE/RMSE for the model and a last-observation
baseline, row counts and the split boundary. Use a fresh output directory to
preserve earlier results.

## Verification and limitations

Four automated tests cover causal features, unchanged targets, invalid input
and a synthetic end-to-end fit/export. They do not establish performance on the
official Acea datasets. No real-data benchmark or improvement claim is made.

The fixed model has not been tuned per dataset. Source publication delays,
irregular sampling and station-specific missing-data behavior need review for
any real operational forecast. The 8 October 2026 repair was made with OpenAI
Codex assistance; it is a correction to the public research code, not a new
historical client result.
