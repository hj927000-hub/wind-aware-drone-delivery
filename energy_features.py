"""Route geometry inputs and an energy-per-metre model for Days 12–13.

Inputs are available from the planned path, before running the energy simulator.
No flight time, energy, SoC or simulator output is used as a prediction feature.
"""
import json
from math import hypot

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from day10_train_energy_model import FEATURES

DIRECTIONS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
HEADING_FEATURES = ['fraction_'+name for name in ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']]
RATE_FEATURES = FEATURES[1:] + HEADING_FEATURES
ROUTE_FEATURES = FEATURES + HEADING_FEATURES


def heading_fractions(path):
    """Length shares of eight OUTBOUND headings, with diagonal length sqrt(2)."""
    lengths = np.zeros(8)
    for a, b in zip(path, path[1:]):
        step = (b[0]-a[0], b[1]-a[1])
        if step not in DIRECTIONS:
            raise ValueError('Path must contain adjacent, distinct grid cells.')
        lengths[DIRECTIONS.index(step)] += hypot(*step)
    if lengths.sum() == 0:
        raise ValueError('Path must contain at least one segment.')
    return lengths / lengths.sum()


def add_route_features(frame, routes_path):
    """Join geometry by route ID and reject inconsistent endpoints or distances."""
    document = json.loads(routes_path.read_text())
    lookup = {}
    cell = document['cell_m']
    for route in document['routes']:
        key, path = route['route_id'], route['path']
        if key in lookup:
            raise ValueError('Duplicate route IDs in route document.')
        if path[0] != document['hub_cell'] or path[-1] != route['customer']:
            raise ValueError('Path endpoints do not match route metadata.')
        length = 2*sum(hypot(b[0]-a[0], b[1]-a[1])*cell for a, b in zip(path, path[1:]))
        if not np.isclose(length, route['planned_round_trip_m']):
            raise ValueError('Route length differs from path geometry.')
        lookup[key] = (route, heading_fractions(path))
    result = frame.copy()
    for key, group in frame.groupby('route_id'):
        if key not in lookup:
            raise ValueError(f'Missing route geometry: {key}')
        route, fractions = lookup[key]
        expected = [*((np.asarray(document['hub_cell'])+.5)*cell),
                    *((np.asarray(route['customer'])+.5)*cell)]
        columns = ['hub_x_m', 'hub_y_m', 'customer_x_m', 'customer_y_m']
        if not np.allclose(group[columns], expected) or not np.allclose(
                group.planned_round_trip_m, route['planned_round_trip_m']):
            raise ValueError('Mission and route geometry disagree.')
        result.loc[group.index, HEADING_FEATURES] = np.tile(fractions, (len(group), 1))
    return result


class EnergyPerMetreForest(RegressorMixin, BaseEstimator):
    """Learn Wh/m from weather, payload and path shape; multiply by planned metres.

    Scale invariance is an explicit modeling assumption for this cruise-only task.
    It will need revisiting when adding fixed takeoff, landing or hover energy.
    """
    def __init__(self, n_estimators=200, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        distance = np.asarray(X['planned_round_trip_m'], dtype=float)
        target = np.asarray(y, dtype=float)
        if not np.isfinite(distance).all() or (distance <= 0).any():
            raise ValueError('Planned distance must be finite and positive.')
        if not np.isfinite(target).all() or (target <= 0).any():
            raise ValueError('Energy labels must be finite and positive.')
        self.forest_ = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=12,
                                            min_samples_leaf=2, random_state=self.random_state, n_jobs=-1)
        self.forest_.fit(X[RATE_FEATURES], target/distance)
        return self

    def predict(self, X):
        check_is_fitted(self, 'forest_')
        distance = np.asarray(X['planned_round_trip_m'], dtype=float)
        if not np.isfinite(distance).all() or (distance <= 0).any():
            raise ValueError('Planned distance must be finite and positive.')
        return self.forest_.predict(X[RATE_FEATURES])*distance


def predict_feasible(model, frame, features):
    """Honor the simulator feasibility flag; ML cannot turn a rejected trip into a flight."""
    if not frame.feasible.isin([0, 1]).all():
        raise ValueError('Invalid feasibility flags.')
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = frame.feasible == 1
    if valid.any():
        result.loc[valid] = model.predict(frame.loc[valid, features])
    return result
