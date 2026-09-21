"""Day 10: first supervised ML energy models, tested on unseen destinations.

X is available AFTER distance-based route planning but BEFORE energy simulation.
y is simulated cruise energy in Wh. No model makes flight-feasibility decisions.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from day9_generate_dataset import sha256_file

ROOT = Path(__file__).resolve().parent
# Explicit allowlist: energy, flight time, SoC and scenario/route IDs stay OUT of X.
FEATURES = ['planned_round_trip_m', 'wind_speed_mps', 'wind_u_mps', 'wind_v_mps', 'payload_kg']
TARGET = 'energy_wh'


def load_dataset(path):
    frame = pd.read_csv(path)
    required = FEATURES + [TARGET, 'feasible', 'route_id', 'scenario_id',
                           'hub_x_m', 'hub_y_m', 'customer_x_m', 'customer_y_m']
    missing = sorted(set(required)-set(frame.columns))
    if missing:
        raise ValueError(f'Missing dataset columns: {missing}')
    if not frame.feasible.isin([0, 1]).all() or frame.scenario_id.duplicated().any():
        raise ValueError('Invalid feasibility flags or duplicate scenario IDs.')
    usable = frame.loc[frame.feasible == 1].copy().reset_index(drop=True)
    if usable.route_id.isna().any() or usable.empty:
        raise ValueError('No usable labeled routes.')
    if not np.isfinite(usable[FEATURES+[TARGET]].to_numpy(dtype=float)).all():
        raise ValueError('Feasible rows must have finite numeric features and labels.')
    if (usable[TARGET] <= 0).any() or (usable.planned_round_trip_m <= 0).any():
        raise ValueError('Positive route lengths and energy labels are required.')
    # A delivery endpoint pair, rather than an arbitrary row ID, is the split unit.
    endpoint_cols = ['hub_x_m', 'hub_y_m', 'customer_x_m', 'customer_y_m']
    usable['delivery_group'] = usable[endpoint_cols].astype(str).agg('|'.join, axis=1)
    if usable.delivery_group.nunique() < 4:
        raise ValueError('At least four different endpoint pairs are needed for a group holdout.')
    if usable.groupby('route_id').delivery_group.nunique().max() != 1:
        raise ValueError('A route ID must refer to one endpoint pair.')
    return usable, len(frame)-len(usable)


def split_by_destination(frame, seed=42, test_size=.25):
    """Keep ALL weather/payload variants of one delivery in the same partition."""
    if not 0 < test_size < 1:
        raise ValueError('test_size must be between zero and one.')
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(frame[FEATURES], frame[TARGET], groups=frame.delivery_group))
    if set(frame.iloc[train].delivery_group) & set(frame.iloc[test].delivery_group):
        raise RuntimeError('Delivery group leakage.')
    return train, test


def scores(truth, prediction):
    return {'mae_wh': float(mean_absolute_error(truth, prediction)),
            'rmse_wh': float(np.sqrt(mean_squared_error(truth, prediction))),
            'r2': float(r2_score(truth, prediction)),
            'negative_predictions': int(np.sum(np.asarray(prediction) < 0))}


def plot_predictions(truth, predictions, metrics, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout='constrained')
    for ax, key, name in zip(axes, ['linear_regression', 'random_forest'], ['Linear regression', 'Random Forest']):
        pred = predictions[key]
        low = min(0, float(np.min(pred)), float(np.min(truth)))
        high = max(float(np.max(pred)), float(np.max(truth)))*1.04
        ax.scatter(truth, pred, s=12, alpha=.4, color='#2473b7')
        ax.plot([low, high], [low, high], '--', color='#667085', linewidth=1, label='Perfect prediction')
        ax.set(xlim=(low, high), ylim=(low, high), aspect='equal',
               xlabel='Simulator energy [Wh]', ylabel='ML prediction [Wh]',
               title=f"{name}\nHeld-out MAE: {metrics[key]['test']['mae_wh']:.2f} Wh")
        ax.legend(loc='upper left', fontsize=8)
    fig.suptitle('Day 10 | Unseen destinations on the same Manhattan map')
    fig.supxlabel('Labels are simulated cruise energy, not measured flight data.', fontsize=9)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def train_models(args):
    frame, excluded = load_dataset(args.dataset)
    manifest_path = args.dataset.with_name('dataset_manifest.json')
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['dataset_sha256'] != sha256_file(args.dataset):
            raise ValueError('Dataset differs from its manifest. Regenerate Day 9 or use a new dataset directory.')
    if args.trees < 1:
        raise ValueError('trees must be positive.')
    train_idx, test_idx = split_by_destination(frame, args.seed)
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    models = {
        'mean_baseline': DummyRegressor(strategy='mean'),
        'linear_regression': make_pipeline(StandardScaler(), LinearRegression()),
        'random_forest': RandomForestRegressor(n_estimators=args.trees, max_depth=12,
                                               min_samples_leaf=2, random_state=args.seed, n_jobs=-1),
    }
    metrics, predictions = {}, {}
    for name, model in models.items():
        # All learned parameters, including scaling, are fitted on training data only.
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        metrics[name] = {'train': scores(y_train, model.predict(X_train)),
                         'test': scores(y_test, predictions[name])}
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'target': TARGET, 'target_unit': 'Wh', 'features': FEATURES,
              'prediction_stage': 'after distance route planning; before energy simulation',
              'split': '25% held-out endpoint-pair groups, not random rows',
              'seed': args.seed, 'train_rows': len(train), 'test_rows': len(test),
              'train_routes': sorted(train.route_id.unique()), 'test_routes': sorted(test.route_id.unique()),
              'excluded_infeasible_rows': excluded, 'models': metrics,
              'forest_parameters': {'n_estimators': args.trees, 'max_depth': 12, 'min_samples_leaf': 2},
              'versions': {'scikit_learn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
              'dataset_sha256': sha256_file(args.dataset),
              'scope': 'Synthetic uniform wind, same map/hub/aircraft/grid/policy; no real-flight validation',
              'interpretation': 'The small fixed holdout is a learning evaluation, not a robust performance guarantee.'}
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    detail = test[['scenario_id', 'route_id', *FEATURES, TARGET]].copy()
    for name, prediction in predictions.items():
        detail[name+'_prediction_wh'] = prediction
        detail[name+'_absolute_error_wh'] = np.abs(prediction-y_test.to_numpy())
    detail.to_csv(args.output/'test_predictions.csv', index=False)
    split = frame[['scenario_id', 'route_id', 'delivery_group']].copy()
    split['partition'] = 'train'
    split.loc[test_idx, 'partition'] = 'test'
    split.to_csv(args.output/'split_membership.csv', index=False)
    for name in ['linear_regression', 'random_forest']:
        # Locally generated model files. Reload with the same scikit-learn version.
        joblib.dump({'model': models[name], 'features': FEATURES, 'target': TARGET,
                     'sklearn_version': sklearn.__version__, 'dataset_sha256': report['dataset_sha256']},
                    args.output/(name+'.joblib'), compress=3)
    plot_predictions(y_test.to_numpy(), predictions, metrics, args.output/'prediction_comparison.png')
    print(f'Train: {len(train)} rows / {train.route_id.nunique()} destinations')
    print(f'Test:  {len(test)} rows / {test.route_id.nunique()} unseen destinations')
    print(f'Excluded infeasible rows: {excluded}')
    for name, result in metrics.items():
        print(f"{name:20s} | train MAE {result['train']['mae_wh']:7.2f} Wh | test MAE {result['test']['mae_wh']:7.2f} Wh | R2 {result['test']['r2']:.3f}")
    example = detail.iloc[0]
    print(f"Example {example.scenario_id}: simulator {example.energy_wh:.2f} Wh, RF {example.random_forest_prediction_wh:.2f} Wh")
    print('Output:', args.output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'outputs/day9/missions.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/day10')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trees', type=int, default=200)
    args = parser.parse_args()
    try:
        train_models(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
