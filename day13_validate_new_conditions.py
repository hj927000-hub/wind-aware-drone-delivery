"""Day 13: evaluate frozen models on new destinations and synthetic conditions.

This script contains no model.fit call. The evaluation protocol is fixed below.
Changing the model after seeing these results makes this a development set.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from day9_generate_dataset import generate, sha256_file
from day10_train_energy_model import load_dataset, scores
from energy_features import add_route_features, predict_feasible

ROOT = Path(__file__).resolve().parent


def validate(args):
    model_hash = sha256_file(args.model)
    bundle = joblib.load(args.model)  # A locally trained artifact, not a downloaded pickle.
    if sha256_file(args.original_routes) != bundle['routes_sha256']:
        raise ValueError('Original routes differ from model training provenance.')
    if sha256_file(args.data_dir/'manhattan_buildings.geojson') != bundle['map_geojson_sha256']:
        raise ValueError('Map differs from model training provenance.')
    if bundle['training_wind_range_mps'] != [0, 9]:
        raise ValueError('This validation protocol expects the default 0–9 m/s training range.')
    original = json.loads(args.original_routes.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'model_sha256_before': model_hash,
        'selected_model_before_evaluation': bundle['selected_model'],
        'destination_count': 12, 'destination_seed': 20260920,
        'exclude_all_original_destinations': [r['customer'] for r in original['routes']],
        'speeds_mps': [0, 1.5, 4.5, 7.5, 9.5, 11],
        'directions_to_deg': list(np.arange(22.5, 360, 45)), 'payloads_kg': [.375, 1.125],
        'interpretation': 'New endpoint pairs on the SAME map/hub/aircraft; paths can share street segments.',
        'stress_case': '9.5 m/s exceeds training maximum 9 m/s but stays below modeled limit 10 m/s.',
        'infeasible_case': '11 m/s: reject using simulator rules; do not issue an ML energy prediction.',
    }
    # Written before generating labels or making predictions.
    (args.output/'validation_protocol.json').write_text(json.dumps(protocol, indent=2))
    generate(argparse.Namespace(
        data_dir=args.data_dir, output=args.output/'dataset', destinations=12, seed=20260920,
        hub=tuple(original['hub_cell']), exclude_customers=protocol['exclude_all_original_destinations'],
        speeds=protocol['speeds_mps'], directions=protocol['directions_to_deg'], payloads=protocol['payloads_kg']))
    csv_path = args.output/'dataset/missions.csv'
    frame, excluded = load_dataset(csv_path)
    if set(frame.delivery_group) & set(bundle['all_original_groups']):
        raise RuntimeError('Validation endpoints overlap the original dataset.')
    all_rows = add_route_features(pd.read_csv(csv_path), args.output/'dataset/routes.json')
    all_rows['regime'] = np.select([
        all_rows.wind_speed_mps == 0,
        all_rows.wind_speed_mps <= 9,
        all_rows.wind_speed_mps <= 10], ['calm_control', 'unseen_conditions_1.5_to_7.5', 'wind_stress_9.5'],
        default='infeasible_11')
    metrics = []
    for name, item in bundle['models'].items():
        prediction = predict_feasible(item['model'], all_rows, item['features'])
        all_rows[name+'_prediction_wh'] = prediction
        valid = all_rows.feasible == 1
        for regime, group in all_rows.loc[valid].groupby('regime'):
            metrics.append({'regime': regime, 'model': name, 'rows': len(group),
                            **scores(group.energy_wh, group[name+'_prediction_wh'])})
        metrics.append({'regime': 'all_feasible', 'model': name, 'rows': int(valid.sum()),
                        **scores(all_rows.loc[valid].energy_wh, prediction[valid])})
    if sha256_file(args.model) != model_hash:
        raise RuntimeError('Model file changed during validation.')
    summary = {'selected_model': bundle['selected_model'], 'rows': len(all_rows),
               'feasible_rows': len(frame), 'infeasible_rows': excluded,
               'new_destinations': frame.delivery_group.nunique(), 'original_endpoint_overlap': 0,
               'model_sha256_before': model_hash, 'model_sha256_after': sha256_file(args.model),
               'validation_dataset_sha256': sha256_file(csv_path),
               'metrics': metrics,
               'note': 'No retraining or reselection; simulated labels, same city, no real-flight validation.'}
    (args.output/'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    pd.DataFrame(metrics).to_csv(args.output/'metrics.csv', index=False)
    all_rows.to_csv(args.output/'predictions.csv', index=False)
    plot_validation(all_rows, pd.DataFrame(metrics), args.output/'new_conditions.png')
    print(f'Day 13: {len(frame)} feasible, {excluded} blocked; 12 new destinations, zero endpoint overlap')
    for row in metrics:
        print(f"{row['regime']:30s} {row['model']:14s} MAE {row['mae_wh']:.3f} Wh")
    return summary


def plot_validation(frame, metrics, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['baseline_rf', 'route_rate_rf']
    labels = ['Original RF', 'Route + Wh/m RF']
    colors = ['#718096', '#0b8a82']
    regimes = ['calm_control', 'unseen_conditions_1.5_to_7.5', 'wind_stress_9.5']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout='constrained')
    x = np.arange(3)
    for index, (name, label, color) in enumerate(zip(names, labels, colors)):
        values = [float(metrics.loc[(metrics.model == name) & (metrics.regime == r), 'mae_wh'].iloc[0]) for r in regimes]
        bars = axes[0].bar(x+(index-.5)*.36, values, .36, label=label, color=color)
        axes[0].bar_label(bars, fmt='%.2f', fontsize=9, padding=3)
        valid = frame.feasible == 1
        grouped = frame.loc[valid].assign(error=frame.loc[valid, name+'_prediction_wh']-frame.loc[valid, 'energy_wh']).groupby('wind_speed_mps').error.mean()
        axes[1].plot(grouped.index, grouped.values, 'o-', label=label, color=color)
    axes[0].set(xticks=x, xticklabels=['Calm\n24 rows', 'New conditions\n576 rows', '9.5 m/s stress\n192 rows'],
                ylabel='Mean absolute error [Wh]', title='Frozen models: 12 new destinations')
    axes[0].margins(y=.2)
    axes[0].legend()
    axes[1].axhline(0, color='#667085', linestyle='--')
    axes[1].axvspan(9, 10, color='#f6ddb0', alpha=.45)
    axes[1].set(xlabel='Wind speed [m/s]', ylabel='Mean prediction minus simulator [Wh]',
                title='Beyond training wind: underestimation risk')
    fig.suptitle('Day 13 | New synthetic conditions; no retraining')
    fig.supxlabel('Training wind: 0, 3, 6, 9 m/s. 11 m/s: 192 rejected cases, no ML prediction. Simulator labels only.', fontsize=9)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=ROOT/'outputs/day12/models.joblib')
    p.add_argument('--original-routes', type=Path, default=ROOT/'outputs/day9/routes.json')
    p.add_argument('--data-dir', type=Path, default=ROOT/'data')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/day13')
    validate(p.parse_args())


if __name__ == '__main__':
    main()
