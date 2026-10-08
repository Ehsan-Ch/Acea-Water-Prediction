"""Acea forecasting with past-only features and a chronological holdout."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

EXPECTED_FILES = [
    "Aquifer_Auser.csv",
    "Aquifer_Petrignano.csv",
    "Aquifer_Doganella.csv",
    "Aquifer_Luco.csv",
    "Water_Spring_Amiata.csv",
    "Water_Spring_Madonna_di_Canneto.csv",
    "Water_Spring_Lupa.csv",
    "River_Arno.csv",
    "Lake_Bilancino.csv",
]

# Dataset configuration
DATASETS_CFG = {
    "Aquifer_Auser.csv": {
        "type": "aquifer",
        "rename_map": {
            "Date": "date",
            "Rainfall_Gallicano": "rainfall_gallicano",
            "Rainfall_Pontetetto": "rainfall_pontetetto",
            "Rainfall_Monte_Serra": "rainfall_monteserra",
            "Rainfall_Orentano": "rainfall_orentano",
            "Rainfall_Borgo_a_Mozzano": "rainfall_borgo",
            "Rainfall_Piaggione": "rainfall_piaggione",
            "Rainfall_Calavorno": "rainfall_calavorno",
            "Rainfall_Croce_Arcana": "rainfall_crocearcana",
            "Rainfall_Tereglio_Coreglia_Antelminelli": "rainfall_tereglio",
            "Rainfall_Fabbriche_di_Vallico": "rainfall_vallico",
            "Depth_to_Groundwater_SAL": "depth_to_groundwater",
            "Depth_to_Groundwater_LT2": "well_lt2",
            "Depth_to_Groundwater_PAG": "well_pag",
            "Depth_to_Groundwater_CoS": "well_cos",
            "Depth_to_Groundwater_DIEC": "well_diec",
            "Temperature_Orentano": "temp_orentano",
            "Temperature_Monte_Serra": "temp_monteserra",
            "Temperature_Ponte_a_Moriano": "temp_moriano",
            "Temperature_Lucca_Orto_Botanico": "temp_lucca",
            "Volume_POL": "volume_pol",
            "Volume_CC1": "volume_cc1",
            "Volume_CC2": "volume_cc2",
            "Volume_CSA": "volume_csa",
            "Volume_CSAL": "volume_csal",
            "Hydrometry_Monte_S_Quirico": "hydrometry_quirico",
            "Hydrometry_Piaggione": "hydrometry_piaggione",
        },
        "target": "depth_to_groundwater",
        "require_cols": ["date", "depth_to_groundwater"],
        "use_cols": [
            "date", "depth_to_groundwater",
            "rainfall_gallicano", "rainfall_pontetetto", "rainfall_monteserra",
            "rainfall_orentano", "rainfall_borgo", "rainfall_piaggione",
            "rainfall_calavorno", "rainfall_crocearcana", "rainfall_tereglio", "rainfall_vallico",
            "temp_orentano", "temp_monteserra", "temp_moriano", "temp_lucca",
            "volume_pol", "volume_cc1", "volume_cc2", "volume_csa", "volume_csal",
            "hydrometry_quirico", "hydrometry_piaggione",
            "well_lt2", "well_pag", "well_cos", "well_diec"
        ],
    },
    "Aquifer_Petrignano.csv": {
        "type": "aquifer",
        "rename_map": {
            "Date": "date",
            "Rainfall_Bastia_Umbra": "rainfall",
            "Depth_to_Groundwater_P24": "depth_p24",
            "Depth_to_Groundwater_P25": "depth_to_groundwater",
            "Temperature_Bastia_Umbra": "temperature",
            "Temperature_Petrignano": "temperature_petrignano",
            "Volume_C10_Petrignano": "drainage_volume",
            "Hydrometry_Fiume_Chiascio_Petrignano": "river_hydrometry",
        },
        "target": "depth_to_groundwater",
        "require_cols": ["date", "depth_to_groundwater"],
        "use_cols": ["date", "rainfall", "temperature", "drainage_volume", "river_hydrometry", "depth_to_groundwater"],
    },
    "Aquifer_Doganella.csv": {
        "type": "aquifer",
        "rename_map": {
            "Date": "date",
            "Rainfall_Monte_Castello": "rainfall",
            "Temperature_Monte_Castello": "temperature",
            "Depth_to_Groundwater_Pozzo_1": "well_1",
            "Depth_to_Groundwater_Pozzo_2": "well_2",
            "Depth_to_Groundwater_Pozzo_3": "well_3",
            "Depth_to_Groundwater_Pozzo_4": "well_4",
            "Depth_to_Groundwater_Pozzo_5": "well_5",
            "Depth_to_Groundwater_Pozzo_6": "well_6",
            "Depth_to_Groundwater_Pozzo_7": "well_7",
            "Depth_to_Groundwater_Pozzo_8": "well_8",
            "Depth_to_Groundwater_Pozzo_9": "depth_to_groundwater",
            "Volume_Cumulative": "drainage_volume",
        },
        "target": "depth_to_groundwater",
        "require_cols": ["date", "depth_to_groundwater"],
        "use_cols": [
            "date", "rainfall", "temperature", "drainage_volume",
            "well_1", "well_2", "well_3", "well_4", "well_5", "well_6", "well_7", "well_8", "depth_to_groundwater"
        ],
    },
    "Aquifer_Luco.csv": {
        "type": "aquifer",
        "rename_map": {
            "Date": "date",
            "Rainfall_Pieve_di_Santo_Stefano": "rainfall",
            "Temperature_Pieve_di_Santo_Stefano": "temperature",
            "Depth_to_Groundwater_Podere_Casetta": "depth_to_groundwater",
            "Volume_Cumulative": "drainage_volume",
        },
        "target": "depth_to_groundwater",
        "require_cols": ["date", "depth_to_groundwater"],
        "use_cols": ["date", "rainfall", "temperature", "drainage_volume", "depth_to_groundwater"],
    },
    "Water_Spring_Amiata.csv": {
        "type": "spring",
        "rename_map": {
            "Date": "date",
            "Rainfall_Mount_Amiata": "rainfall",
            "Temperature_Mount_Amiata": "temperature",
            "Depth_to_Groundwater_SGA": "depth_to_groundwater",
            "Hydrometry_Albegna": "river_hydrometry",
            "Volume_Cumulative": "drainage_volume",
            "Flow_Rate_Bugnano": "flow_bugnano",
            "Flow_Rate_Arbure": "flow_arbure",
            "Flow_Rate_Ermicciolo": "flow_ermicciolo",
            "Flow_Rate_Galleria_Alta": "flow_galleria_alta",
        },
        "target": "flow_ermicciolo",
        "require_cols": ["date", "flow_ermicciolo"],
        "use_cols": ["date", "rainfall", "temperature", "depth_to_groundwater", "river_hydrometry", "drainage_volume", "flow_ermicciolo"],
    },
    "Water_Spring_Madonna_di_Canneto.csv": {
        "type": "spring",
        "rename_map": {
            "Date": "date",
            "Rainfall_Settefrati": "rainfall",
            "Temperature_Settefrati": "temperature",
            "Flow_Rate_Madonna_di_Canneto": "flow_rate",
        },
        "target": "flow_rate",
        "require_cols": ["date", "flow_rate"],
        "use_cols": ["date", "rainfall", "temperature", "flow_rate"],
    },
    "Water_Spring_Lupa.csv": {
        "type": "spring",
        "rename_map": {
            "Date": "date",
            "Rainfall_Terni": "rainfall",
            "Flow_Rate_Lupa": "flow_rate",
        },
        "target": "flow_rate",
        "require_cols": ["date", "flow_rate"],
        "use_cols": ["date", "rainfall", "flow_rate"],
    },
    "River_Arno.csv": {
        "type": "river",
        "rename_map": {
            "Date": "date",
            "Hydrometry_Nave_di_Rosano": "hydrometry",
        },
        "target": "hydrometry",
        "require_cols": ["date", "hydrometry"],
        "use_cols": ["date", "hydrometry", "rainfall"],
    },
    "Lake_Bilancino.csv": {
        "type": "lake",
        "rename_map": {
            "Date": "date",
            "Rainfall_Mugello": "rainfall",
            "Temperature_Mugello": "temperature",
            "Lake_Level": "lake_level",
            "Lake_Outflow": "lake_outflow",
        },
        "target": "lake_level",
        "require_cols": ["date", "lake_level"],
        "use_cols": ["date", "rainfall", "temperature", "lake_level", "lake_outflow"],
    },
}

# Forecasting uses only observations before each target date.
def prepare_frame(frame, cfg):
    frame = frame.rename(columns={str(c): str(c).strip() for c in frame.columns})
    frame = frame.rename(columns=cfg['rename_map']).copy()
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names after renaming')
    missing = set(cfg['require_cols']) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    frame['date'] = pd.to_datetime(frame['date'], dayfirst=True, errors='raise')
    if frame['date'].isna().any() or frame['date'].duplicated().any():
        raise ValueError('Dates must be nonmissing and unique')
    frame = frame.sort_values('date').reset_index(drop=True)
    columns = [c for c in cfg['use_cols'] if c in frame and c != 'date']
    for col in columns:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
    frame = frame[['date'] + columns].replace([np.inf, -np.inf], np.nan)
    return frame


def build_features(frame, target):
    """Predict the next observed row; current exogenous values are unavailable."""
    features = pd.DataFrame(index=frame.index)
    dates = frame['date']
    features['month_sin'] = np.sin(2 * np.pi * dates.dt.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * dates.dt.month / 12)
    features['day_of_year'] = dates.dt.dayofyear
    for col in frame.select_dtypes(include='number').columns:
        # Forward filling uses only previously observed values, never future rows.
        past = frame[col].ffill().shift(1)
        for lag in (1, 3, 7, 14, 30):
            features[f'{col}_lag_{lag}'] = past.shift(lag - 1)
        for window in (3, 7, 14, 30):
            features[f'{col}_mean_{window}'] = past.rolling(window, min_periods=1).mean()
    if f'{target}_lag_1' not in features:
        raise ValueError(f'Target {target!r} is not numeric')
    return features


def run_one(name, cfg, path, output=None, min_train=365):
    """Fixed model and chronological holdout, with training-only imputation."""
    frame = prepare_frame(pd.read_csv(path), cfg)
    features = build_features(frame, cfg['target'])
    cutoff = int(len(frame) * 0.8)
    if cutoff < min_train or cutoff >= len(frame):
        raise ValueError(f'Need at least {min_train} training rows and a holdout')
    target = frame[cfg['target']]
    baseline = features[f"{cfg['target']}_lag_1"]
    train = (frame.index < cutoff) & target.notna() & baseline.notna()
    test = (frame.index >= cutoff) & target.notna() & baseline.notna()
    if train.sum() < min_train - 1 or test.sum() < 1:
        raise ValueError('Insufficient observed targets after missing-value checks')
    model = Pipeline([
        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('model', GradientBoostingRegressor(random_state=42)),
    ])
    model.fit(features.loc[train], target.loc[train])
    prediction = model.predict(features.loc[test])
    observed = target.loc[test]
    report = {
        'dataset': name, 'target': cfg['target'], 'train_rows': int(train.sum()),
        'test_rows': int(test.sum()), 'train_end': str(frame.loc[train, 'date'].max().date()),
        'test_start': str(frame.loc[test, 'date'].min().date()),
        'mae': float(mean_absolute_error(observed, prediction)),
        'rmse': float(np.sqrt(mean_squared_error(observed, prediction))),
        'baseline_mae': float(mean_absolute_error(observed, baseline.loc[test])),
        'baseline_rmse': float(np.sqrt(mean_squared_error(observed, baseline.loc[test]))),
        'protocol': 'Sequential next-observation forecasting; fixed model; 80/20 chronological split',
    }
    if output is not None:
        destination = Path(output) / Path(name).stem
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        pd.DataFrame({'date': frame.loc[test, 'date'], 'actual': observed,
                      'prediction': prediction, 'last_observed': baseline.loc[test]}).to_csv(
                          destination / 'predictions.csv', index=False)
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Acea next-observation forecasting')
    parser.add_argument('--data', type=Path, required=True, help='Directory containing official CSV files')
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    args = parser.parse_args(argv)
    found = [(name, cfg, args.data / name) for name, cfg in DATASETS_CFG.items()
             if (args.data / name).is_file()]
    if not found:
        parser.error('No supported Acea CSV files found in --data')
    failures = []
    for name, cfg, path in found:
        try:
            run_one(name, cfg, path, output=args.output)
        except (ValueError, OSError) as exc:
            failures.append(name)
            print(f'{name}: {exc}', file=sys.stderr)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
