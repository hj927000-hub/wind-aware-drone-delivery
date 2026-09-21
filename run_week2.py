"""Day 14: reproduce the complete Week 2 experiment offline using bundled OSM.

python3 run_week2.py
python3 run_week2.py --output /path/to/new/results
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'outputs')
    p.add_argument('--skip-tests', action='store_true', help='Skip the final test gate; recorded in the report.')
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    commands = [
        ['day8_manhattan.py', '--preview-3d', '--output', out/'day8'],
        ['day9_generate_dataset.py', '--output', out/'day9'],
        ['day10_train_energy_model.py', '--dataset', out/'day9/missions.csv', '--output', out/'day10'],
        ['day11_analyze_errors.py', '--predictions', out/'day10/test_predictions.csv', '--output', out/'day11'],
        ['day12_improve_model.py', '--dataset', out/'day9/missions.csv', '--routes', out/'day9/routes.json', '--output', out/'day12'],
        ['day13_validate_new_conditions.py', '--model', out/'day12/models.joblib', '--original-routes', out/'day9/routes.json', '--output', out/'day13'],
    ]
    executed = []
    for command in commands:
        print('\nRunning:', command[0], flush=True)
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, check=True)
        executed.append(command[0])
    if not args.skip_tests:
        subprocess.run([sys.executable, '-m', 'unittest', 'test_day56', 'test_day8', 'test_day910', 'test_week2'],
                       cwd=ROOT, check=True)
    from day9_generate_dataset import sha256_file
    report = {'stages_completed': executed, 'tests': 'skipped' if args.skip_tests else 'passed',
              'python_version': platform.python_version(), 'elapsed_seconds': round(time.monotonic()-start, 2),
              'data_sha256': sha256_file(out/'day9/missions.csv'),
              'validation_sha256': sha256_file(out/'day13/dataset/missions.csv'),
              'code_sha256': {f.name: sha256_file(f) for f in sorted(ROOT.glob('*.py'))},
              'scope': 'Actual OSM footprints; synthetic weather and simulator-generated energy labels.'}
    (out/'week2_run.json').write_text(json.dumps(report, indent=2))
    print('\nWeek 2 complete. Results:', out)


if __name__ == '__main__':
    main()
