"""Day 11: inspect the already-viewed Day 10 holdout without refitting any model."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = ['linear_regression', 'random_forest']


def analyze(args):
    frame = pd.read_csv(args.predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    tables = {}
    for grouping in ['wind_speed_mps', 'route_id', 'payload_kg']:
        rows = []
        for value, group in frame.groupby(grouping):
            for model in MODELS:
                error = group[model+'_prediction_wh'] - group.energy_wh
                rows.append({grouping: value, 'model': model, 'rows': len(group),
                             'mae_wh': float(error.abs().mean()),
                             'bias_wh': float(error.mean()),
                             'max_absolute_error_wh': float(error.abs().max())})
        tables[grouping] = pd.DataFrame(rows)
        tables[grouping].to_csv(args.output/f'errors_by_{grouping}.csv', index=False)
    worst = frame.nlargest(10, 'random_forest_absolute_error_wh')
    worst.to_csv(args.output/'worst_predictions.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    for model, color, label in zip(MODELS, ['#b75d36', '#187d98'], ['Linear regression', 'Random Forest']):
        table = tables['wind_speed_mps'].query('model == @model')
        axes[0].plot(table.wind_speed_mps, table.mae_wh, 'o-', label=label, color=color)
    axes[0].set(xlabel='Uniform wind speed [m/s]', ylabel='Mean absolute error [Wh]', title='Where are errors larger?')
    axes[0].legend()
    error = frame.random_forest_prediction_wh-frame.energy_wh
    axes[1].scatter(frame.planned_round_trip_m, error, c=frame.wind_speed_mps, cmap='viridis', s=16, alpha=.5)
    axes[1].axhline(0, color='#667085', linestyle='--')
    axes[1].set(xlabel='Planned round-trip distance [m]', ylabel='RF prediction minus simulator [Wh]',
                title='Positive = overestimate; negative = underestimate')
    fig.suptitle('Day 11 | Diagnose the original 450-row holdout')
    fig.supxlabel('Exploratory diagnosis of simulator labels; this is not a fresh validation set.', fontsize=9)
    fig.savefig(args.output/'error_analysis.png', dpi=150)
    plt.close(fig)
    report = {'rows': len(frame), 'role': 'Exploratory analysis; holdout already inspected in Day 10',
              'rf_worst_scenario': str(worst.iloc[0].scenario_id),
              'rf_max_error_wh': float(worst.iloc[0].random_forest_absolute_error_wh),
              'linear_negative_predictions': int((frame.linear_regression_prediction_wh < 0).sum()),
              'rf_mae_by_wind': tables['wind_speed_mps'].query("model == 'random_forest'").to_dict('records')}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"Day 11: {len(frame)} rows; largest RF error {report['rf_max_error_wh']:.2f} Wh")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, default=ROOT/'outputs/day10/test_predictions.csv')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/day11')
    analyze(p.parse_args())


if __name__ == '__main__':
    main()
