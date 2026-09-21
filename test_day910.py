"""Checks for physical labels, infeasibility handling and group leakage."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from day4_wind_field import WindConfig, build_wind_field
from day5_wind_energy_model import FlightConfig
from day8_manhattan import RealCity
from day9_generate_dataset import simulate_trip, weather_cases
from day10_train_energy_model import FEATURES, load_dataset, split_by_destination


class Day910Tests(unittest.TestCase):
    def setUp(self):
        self.city = RealCity(np.zeros((4, 4), dtype=bool), 10)
        self.route = {'path': [(0, 0), (1, 0), (2, 0)]}

    def test_round_trip_label_matches_independent_calm_calculation(self):
        field = build_wind_field(self.city, WindConfig(0, 0, 6, False))
        result = simulate_trip(self.city, field, self.route, 1.5, FlightConfig())
        # 20 m out carrying 1.5 kg, 20 m back empty; both at 12 m/s.
        expected = (1430+1250)*(20/12)/3600
        self.assertAlmostEqual(result['energy_wh'], expected)
        self.assertAlmostEqual(result['round_trip_m'], 40)
        self.assertTrue(result['feasible'])

    def test_prohibited_wind_has_no_energy_label(self):
        field = build_wind_field(self.city, WindConfig(11, 0, 6, False))
        result = simulate_trip(self.city, field, self.route, 1.5, FlightConfig())
        self.assertFalse(result['feasible'])
        self.assertNotIn('energy_wh', result)

    def test_calm_weather_is_not_duplicated_by_direction(self):
        cases = list(weather_cases([0, 0, 3], [0, 90, 360]))
        self.assertEqual(sum(speed == 0 for speed, _ in cases), 1)
        self.assertEqual(len(cases), 3)

    def test_endpoint_groups_do_not_leak_even_with_route_aliases(self):
        rows = []
        for group in range(8):
            for variant in range(3):
                rows.append({**{feature: 1.0 for feature in FEATURES},
                             'energy_wh': 10+group, 'delivery_group': f'endpoint-{group}',
                             'route_id': f'alias-{group}-{variant}'})
        frame = pd.DataFrame(rows)
        train, test = split_by_destination(frame)
        self.assertFalse(set(frame.iloc[train].delivery_group) & set(frame.iloc[test].delivery_group))
        self.assertEqual(len(train)+len(test), len(frame))

    def test_invalid_feasible_label_is_rejected_and_failures_are_excluded(self):
        rows = []
        for group in range(4):
            rows.append({**{feature: 1.0 for feature in FEATURES}, 'energy_wh': 10.0,
                         'feasible': 1, 'scenario_id': f'S{group}', 'route_id': f'R{group}',
                         'hub_x_m': 0, 'hub_y_m': 0, 'customer_x_m': 10+group, 'customer_y_m': 10})
        rows.append({**rows[0], 'scenario_id': 'failed', 'feasible': 0, 'energy_wh': np.nan})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'missions.csv'
            frame = pd.DataFrame(rows)
            frame.to_csv(path, index=False)
            usable, excluded = load_dataset(path)
            self.assertEqual((len(usable), excluded), (4, 1))
            frame.loc[0, 'energy_wh'] = np.nan
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'finite numeric'):
                load_dataset(path)


if __name__ == '__main__':
    unittest.main()
