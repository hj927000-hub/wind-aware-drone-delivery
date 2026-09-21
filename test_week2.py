"""Week 2 guards: geometry features, scaling, blocked predictions and CV separation."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from energy_features import (EnergyPerMetreForest, HEADING_FEATURES, ROUTE_FEATURES,
                             add_route_features, heading_fractions, predict_feasible)
from day10_train_energy_model import split_by_destination


class Week2Tests(unittest.TestCase):
    def test_direction_features_are_length_weighted_and_outbound(self):
        shares = heading_fractions([(0, 0), (1, 0), (2, 1)])
        self.assertAlmostEqual(shares[0], 1/(1+np.sqrt(2)))
        self.assertAlmostEqual(shares[1], np.sqrt(2)/(1+np.sqrt(2)))
        self.assertAlmostEqual(sum(shares), 1)
        reverse = heading_fractions([(2, 1), (1, 0), (0, 0)])
        self.assertAlmostEqual(reverse[4], shares[0])
        self.assertAlmostEqual(reverse[5], shares[1])

    def test_invalid_path_cannot_become_a_feature(self):
        for path in [[(0, 0)], [(0, 0), (2, 1)], [(0, 0), (0, 0)]]:
            with self.assertRaises(ValueError):
                heading_fractions(path)

    def test_rate_model_learns_constant_rate_and_scales_unseen_distance(self):
        X = pd.DataFrame(0., index=range(8), columns=ROUTE_FEATURES)
        X['planned_round_trip_m'] = np.arange(1, 9)*100.
        X['fraction_E'] = 1.
        model = EnergyPerMetreForest(10, 42).fit(X, X.planned_round_trip_m*.03)
        new = X.iloc[:1].copy()
        new['planned_round_trip_m'] = 2000.
        self.assertAlmostEqual(model.predict(new)[0], 60.)
        new['planned_round_trip_m'] = 0
        with self.assertRaises(ValueError):
            model.predict(new)

    def test_infeasible_rows_never_reach_regressor(self):
        class Spy:
            def predict(self, X):
                if not (X['wind_speed_mps'] <= 10).all():
                    raise AssertionError('Blocked trip reached model.')
                return np.full(len(X), 5.)
        data = pd.DataFrame({'feasible': [1, 0], 'wind_speed_mps': [4, 11]})
        result = predict_feasible(Spy(), data, ['wind_speed_mps'])
        self.assertEqual(result.iloc[0], 5)
        self.assertTrue(np.isnan(result.iloc[1]))
        self.assertTrue(predict_feasible(Spy(), data.iloc[1:], ['wind_speed_mps']).isna().all())

    def test_route_join_rejects_mismatched_distance(self):
        document = {'cell_m': 5, 'hub_cell': [0, 0], 'routes': [
            {'route_id': 'R1', 'path': [[0, 0], [1, 0]], 'customer': [1, 0], 'planned_round_trip_m': 10}]}
        frame = pd.DataFrame([{'route_id': 'R1', 'hub_x_m': 2.5, 'hub_y_m': 2.5,
                               'customer_x_m': 7.5, 'customer_y_m': 2.5, 'planned_round_trip_m': 10}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'routes.json'
            path.write_text(json.dumps(document))
            enriched = add_route_features(frame, path)
            self.assertEqual(enriched.fraction_E.iloc[0], 1.)
            frame['planned_round_trip_m'] = 11
            with self.assertRaisesRegex(ValueError, 'disagree'):
                add_route_features(frame, path)

    def test_group_cv_keeps_holdout_out_and_destinations_separate(self):
        frame = pd.DataFrame({name: np.ones(72) for name in ROUTE_FEATURES})
        frame['energy_wh'] = 10.
        frame['delivery_group'] = np.repeat(np.arange(24), 3)
        train_idx, hold_idx = split_by_destination(frame)
        train, hold = frame.iloc[train_idx], frame.iloc[hold_idx]
        for fit_idx, val_idx in GroupKFold(3).split(train, groups=train.delivery_group):
            fit, val = train.iloc[fit_idx], train.iloc[val_idx]
            self.assertFalse(set(fit.delivery_group) & set(val.delivery_group))
            self.assertFalse(set(train.delivery_group) & set(hold.delivery_group))


if __name__ == '__main__':
    unittest.main()
