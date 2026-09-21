"""Day 12: select a fixed candidate using training-only destination-group CV.

The original test set has been inspected: it is reported as exploratory only.
Two predeclared candidates; no tuning against the Day 13 validation data.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from day9_generate_dataset import sha256_file
from day10_train_energy_model import FEATURES, TARGET, load_dataset, scores, split_by_destination
from energy_features import EnergyPerMetreForest, ROUTE_FEATURES, add_route_features

ROOT = Path(__file__).resolve().parent


def train(args):
    manifest = json.loads(args.dataset.with_name('dataset_manifest.json').read_text())
    if sha256_file(args.dataset) != manifest['dataset_sha256']:
        raise ValueError('Training CSV differs from its manifest.')
    frame, excluded = load_dataset(args.dataset)
    frame = add_route_features(frame, args.routes)
    train_idx, test_idx = split_by_destination(frame, args.seed)
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    candidates = {
        'baseline_rf': (RandomForestRegressor(n_estimators=args.trees, max_depth=12,
                        min_samples_leaf=2, random_state=args.seed, n_jobs=-1), FEATURES),
        'route_rate_rf': (EnergyPerMetreForest(args.trees, args.seed), ROUTE_FEATURES),
    }
    folds = list(GroupKFold(n_splits=3).split(train, groups=train.delivery_group))
    cv_rows, fold_rows = [], []
    for fold, (fit_idx, val_idx) in enumerate(folds, 1):
        fit, val = train.iloc[fit_idx], train.iloc[val_idx]
        if set(fit.delivery_group) & set(val.delivery_group):
            raise RuntimeError('Group leakage in CV.')
        for part, data in [('fit', fit), ('validate', val)]:
            fold_rows.extend({'fold': fold, 'partition': part, 'route_id': row.route_id,
                              'delivery_group': row.delivery_group}
                             for row in data[['route_id', 'delivery_group']].drop_duplicates().itertuples())
        for name, (candidate, features) in candidates.items():
            model = clone(candidate).fit(fit[features], fit[TARGET])
            cv_rows.append({'fold': fold, 'model': name, 'rows': len(val),
                            **scores(val[TARGET], model.predict(val[features]))})
    cv = pd.DataFrame(cv_rows)
    means = cv.groupby('model').mae_wh.mean().to_dict()
    selected = min(means, key=means.get)
    args.output.mkdir(parents=True, exist_ok=True)
    cv.to_csv(args.output/'cross_validation.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(args.output/'cv_groups.csv', index=False)
    fitted = {name: {'model': clone(model).fit(train[features], train[TARGET]), 'features': features}
              for name, (model, features) in candidates.items()}
    bundle = {'models': fitted, 'selected_model': selected, 'sklearn_version': sklearn.__version__,
              'dataset_sha256': sha256_file(args.dataset), 'routes_sha256': sha256_file(args.routes),
              'train_groups': sorted(train.delivery_group.unique()),
              'all_original_groups': sorted(frame.delivery_group.unique()),
              'seed': args.seed, 'target': TARGET,
              'map_geojson_sha256': manifest['map_geojson_sha256'],
              'training_wind_range_mps': [float(train.wind_speed_mps.min()), float(train.wind_speed_mps.max())]}
    # Freeze BOTH candidates and the choice before evaluating any test predictions.
    model_path = args.output/'models.joblib'
    joblib.dump(bundle, model_path, compress=3)
    detail = test[['scenario_id', 'route_id', *FEATURES, TARGET]].copy()
    holdout_scores = {}
    for name, item in fitted.items():
        prediction = item['model'].predict(test[item['features']])
        detail[name+'_prediction_wh'] = prediction
        holdout_scores[name] = scores(test[TARGET], prediction)
    detail.to_csv(args.output/'exploratory_holdout_predictions.csv', index=False)
    report = {'selection_rule': 'Lowest mean MAE across 3 destination-group folds of ORIGINAL TRAINING rows only',
              'cv_mean_mae_wh': means, 'selected_model': selected, 'train_rows': len(train),
              'train_destinations': train.delivery_group.nunique(), 'excluded_infeasible_rows': excluded,
              'exploratory_original_holdout': holdout_scores,
              'model_sha256': sha256_file(model_path), 'dataset_sha256': bundle['dataset_sha256'],
              'routes_sha256': bundle['routes_sha256'], 'sklearn_version': sklearn.__version__,
              'feature_source': 'Planned path length and heading shares, synthetic wind, payload; no energy/time/SoC inputs',
              'caveat': 'Normalization and heading features change together; this is not an ablation study.',
              'code_sha256': {f: sha256_file(ROOT/f) for f in ['day12_improve_model.py', 'energy_features.py']}}
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('Day 12 training-only CV MAE [Wh]:', means)
    print('Frozen selection:', selected)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT/'outputs/day9/missions.csv')
    p.add_argument('--routes', type=Path, default=ROOT/'outputs/day9/routes.json')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/day12')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--trees', type=int, default=200)
    args = p.parse_args()
    if args.trees < 1:
        p.error('trees must be positive')
    train(args)


if __name__ == '__main__':
    main()
