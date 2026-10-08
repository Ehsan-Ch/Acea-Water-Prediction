import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from Acea_Water import DATASETS_CFG, build_features, prepare_frame, run_one


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.cfg = DATASETS_CFG['Water_Spring_Lupa.csv']
        n = 100
        self.raw = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=n).strftime('%d/%m/%Y'),
                                 'Flow_Rate_Lupa': 10 + np.sin(np.arange(n) / 8),
                                 'Rainfall_Terni': np.arange(n, dtype=float) % 5})

    def test_current_and_future_measurements_cannot_change_prior_features(self):
        frame = prepare_frame(self.raw, self.cfg)
        changed = frame.copy()
        changed.loc[50:, ['flow_rate', 'rainfall']] = 99999
        pd.testing.assert_frame_equal(build_features(frame, 'flow_rate').loc[:50],
                                      build_features(changed, 'flow_rate').loc[:50])

    def test_positive_target_values_are_not_replaced(self):
        frame = prepare_frame(self.raw, self.cfg)
        np.testing.assert_array_equal(frame.flow_rate, self.raw.Flow_Rate_Lupa)

    def test_required_columns_and_duplicate_dates_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            prepare_frame(self.raw.drop(columns='Flow_Rate_Lupa'), self.cfg)
        self.raw.loc[1, 'Date'] = self.raw.loc[0, 'Date']
        with self.assertRaisesRegex(ValueError, 'unique'):
            prepare_frame(self.raw, self.cfg)

    def test_synthetic_pipeline_exports_model_and_baseline_errors(self):
        self.raw['Rainfall_Terni'] = np.nan
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / 'input.csv'
            self.raw.to_csv(path, index=False)
            report = run_one('Water_Spring_Lupa.csv', self.cfg, path, output=tmp, min_train=40)
            self.assertLess(report['train_end'], report['test_start'])
            self.assertEqual(report['test_rows'], 20)
            self.assertTrue(np.isfinite(report['baseline_rmse']))
            self.assertTrue(np.isfinite(report['rmse']))
            self.assertTrue((Path(tmp) / 'Water_Spring_Lupa/metrics.json').is_file())


if __name__ == '__main__':
    unittest.main()
