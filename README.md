# Acea Water Prediction

A Python hydrological-modelling project based on the Acea Water Prediction challenge. The published script defines a shared experiment structure for nine datasets covering aquifers, springs, a river and a lake.

## Project scope

The code defines dataset-specific column mappings, target variables and a training/evaluation flow. Targets include groundwater depth, river hydrometry, lake level and spring flow rate.

The intended workflow combines data preparation, calendar features, lag/rolling features, chronological splitting and evaluation with mean absolute error (MAE) and root mean squared error (RMSE).

## Current repository status

This is an incomplete code snapshot. Several helper functions referenced by `Acea_Water.py` are not defined or imported in the published script, so it cannot currently run as a standalone pipeline.

The missing helpers are:

- `list_found_files`
- `robust_rename`
- `coalesce_station_columns`
- `parse_and_sort_date`
- `forward_fill_exogenous`
- `clip_outliers`
- `add_time_features`
- `add_lag_rolling`
- `split_train_test`
- `select_numeric_features`
- `choose_model`

The script imports gradient boosting, random forest and ElasticNet estimators. Model-selection behaviour depends on restoring `choose_model`; imports alone do not establish which experiments have been executed.

## Dataset configuration

| Dataset group | Configured datasets |
| --- | --- |
| Aquifers | Auser, Petrignano, Doganella, Luco |
| Springs | Amiata, Madonna di Canneto, Lupa |
| River | Arno |
| Lake | Bilancino |

Expected CSV filenames and column mappings are listed in `EXPECTED_FILES` and `DATASETS_CFG` inside the script. The data is not included.

## Dependencies and paths

The code uses Python, pandas, NumPy and scikit-learn. The configured input paths are `/content/acea-water-prediction` and `/content`; adapt `BASE_PATHS` for a local environment.

```bash
python -m pip install numpy pandas scikit-learn
```

Restore the missing functions before attempting:

```bash
python Acea_Water.py
```

## Evaluation considerations

The script contains MAE/RMSE calculation and reporting logic, but this repository does not include verified benchmark results.

The Lupa-specific routine replaces non-negative flow values with averages for the same day and month across years. This transformation, and any data-dependent preprocessing, should be reviewed against the forecast objective and fitted without using held-out data.

## Next development steps

Restore the helper functions, document the forecast horizon and data assumptions, fit transformations on training data, add a simple baseline, and save reproducible evaluation results with environment and split details.
