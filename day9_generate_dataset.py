"""Day 9: repeat controlled delivery experiments on the Day 8 Manhattan map.

One row = one round trip on a PREPLANNED distance-shortest route.
The route stays fixed while synthetic uniform wind and outbound payload vary.
Labels come from Day 5's simulator, not measured flights.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from math import cos, hypot, isfinite, radians, sin
from pathlib import Path

import numpy as np
from shapely.ops import unary_union

from day1_drone_mission import Drone
from day3_astar_route import astar
from day4_wind_field import WindConfig, build_wind_field
from day5_wind_energy_model import FlightConfig, evaluate, mission_summary
from day8_manhattan import RealCity, largest_component, load_buildings, rasterize

ROOT = Path(__file__).resolve().parent
CELL_M = 5.0
BUFFER_M = 5.0
DEFAULT_HUB = (93, 45)
DEFAULT_SPEEDS = [0, 3, 6, 9, 11]  # 11 exceeds the model's 10 m/s limit.
DEFAULT_DIRECTIONS = list(range(0, 360, 45))  # TO east=0, north=90.
DEFAULT_PAYLOADS = [0, 0.75, 1.5]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weather_cases(speeds, directions):
    """Avoid eight duplicate calm cases: direction is irrelevant at speed 0."""
    if not speeds or not directions:
        raise ValueError('Provide at least one speed and direction.')
    if any(not isfinite(v) or v < 0 for v in speeds):
        raise ValueError('Wind speeds must be finite and nonnegative.')
    if any(not isfinite(v) for v in directions):
        raise ValueError('Wind directions must be finite.')
    for speed in sorted(set(speeds)):
        angles = [0.0] if speed == 0 else sorted({d % 360 for d in directions})
        for direction in angles:
            yield float(speed), float(direction)


def make_routes(data_dir, count=24, seed=42, hub=DEFAULT_HUB, exclude_customers=()):
    """Freeze map/grid/hub, select reproducible distinct destinations, run A*."""
    buildings, map_meta = load_buildings(data_dir)
    geometry = unary_union([g for g, _ in buildings])
    raw = rasterize(geometry, map_meta['side_m'], CELL_M, 0)
    grid = rasterize(geometry, map_meta['side_m'], CELL_M, BUFFER_M)
    blocked = {(int(x), int(y)) for y, x in np.argwhere(grid)}
    city = RealCity(raw, CELL_M, hub=tuple(hub))
    component = largest_component(city, blocked)
    if city.hub not in component:
        raise ValueError('Hub must be in the largest free component. It is never relocated automatically.')
    excluded = {tuple(c) for c in exclude_customers}
    candidates = sorted(c for c in component if c not in excluded
                        and hypot(c[0]-hub[0], c[1]-hub[1])*CELL_M >= 100)
    if not 1 <= count <= len(candidates):
        raise ValueError(f'Destinations must be between 1 and {len(candidates)}.')
    rng = np.random.default_rng(seed)
    chosen = [candidates[int(i)] for i in rng.choice(len(candidates), count, replace=False)]
    routes = []
    for index, customer in enumerate(chosen):
        path = astar(city, city.hub, customer, blocked)
        if not path:
            raise RuntimeError('Connected destination unexpectedly has no path.')
        one_way = sum(hypot(b[0]-a[0], b[1]-a[1])*CELL_M
                      for a, b in zip(path, path[1:]))
        routes.append({'route_id': f'R{index+1:03d}', 'customer': customer,
                       'path': path, 'planned_round_trip_m': 2*one_way})
    return city, grid, routes, map_meta


def simulate_trip(city, field, route, payload, cfg):
    """Evaluate fixed outbound path and its reverse; return payload is 0 kg."""
    outbound = evaluate(city, field, route['path'], payload, cfg)
    inbound = evaluate(city, field, route['path'][::-1], 0.0, cfg)
    return mission_summary(outbound, inbound)


def plot_routes(city, blocked_grid, routes, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')
    side = city.columns*city.cell_m
    category = np.where(city.raw_grid, 2, np.where(blocked_grid, 1, 0))
    ax.imshow(category, origin='lower', extent=(0, side, 0, side),
              cmap=ListedColormap(['#f5f8fb', '#f7d8a8', '#40526a']),
              vmin=0, vmax=2, interpolation='nearest')
    for route in routes:
        x, y = zip(*(city.cell_center_m(c) for c in route['path']))
        ax.plot(x, y, color='#078275', alpha=.3, linewidth=1)
    points = [city.cell_center_m(r['customer']) for r in routes]
    ax.scatter(*zip(*points), color='#b3327e', s=28, label=f'{len(routes)} destinations', zorder=5)
    hx, hy = city.cell_center_m(city.hub)
    ax.scatter(hx, hy, marker='s', s=95, color='#2473b7', label='Fixed hub', zorder=6)
    ax.set(xlabel='Local east [m]', ylabel='Local north [m]', aspect='equal',
           title='Day 9 | Fixed routes for controlled energy experiments')
    ax.legend(loc='upper left')
    fig.supxlabel('5 m cells / 5 m buffer | © OpenStreetMap contributors', fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate(args):
    payloads = sorted(set(args.payloads))
    if not payloads or any(not isfinite(p) or p < 0 for p in payloads):
        raise ValueError('Payloads must be finite and nonnegative.')
    cases = list(weather_cases(args.speeds, args.directions))
    city, grid, routes, map_meta = make_routes(args.data_dir, args.destinations, args.seed, args.hub,
                                             getattr(args, 'exclude_customers', ()))
    cfg = FlightConfig(risk_weight=0, time_weight=0)
    rows = []
    for case_index, (speed, direction) in enumerate(cases):
        field = build_wind_field(city, WindConfig(speed, direction, cfg.soft_wind, False))
        u, v = speed*cos(radians(direction)), speed*sin(radians(direction))
        for route in routes:
            customer = route['customer']
            hx, hy = city.cell_center_m(city.hub)
            cx, cy = city.cell_center_m(customer)
            for payload in payloads:
                result = simulate_trip(city, field, route, payload, cfg)
                reason = result['reason']
                if speed > cfg.max_wind:
                    reason = 'Uniform wind exceeds max_wind_mps'
                rows.append({
                    'scenario_id': f'S{len(rows)+1:05d}', 'route_id': route['route_id'],
                    'hub_x_m': hx, 'hub_y_m': hy, 'customer_x_m': cx, 'customer_y_m': cy,
                    'cell_m': CELL_M, 'buffer_m': BUFFER_M,
                    'planned_round_trip_m': route['planned_round_trip_m'],
                    'wind_speed_mps': speed, 'wind_direction_to_deg': direction,
                    'wind_u_mps': u, 'wind_v_mps': v, 'payload_kg': payload,
                    'feasible': int(result['feasible']), 'reason': reason,
                    'energy_wh': result.get('energy_wh'),
                    'mission_min': result.get('mission_min'),
                    'final_soc_pct': result.get('final_soc_pct'),
                })
        if (case_index+1) % 8 == 0 or case_index+1 == len(cases):
            print(f'Weather cases: {case_index+1}/{len(cases)} | rows: {len(rows)}', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output/'missions.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    route_document = {'hub_cell': city.hub, 'cell_m': CELL_M, 'buffer_m': BUFFER_M,
                      'route_policy': 'distance_shortest_fixed_outbound_reverse_return', 'routes': routes}
    (args.output/'routes.json').write_text(json.dumps(route_document, indent=2))
    manifest = {
        'schema_version': 1, 'label_source': 'Day 5 educational simulator; not measured flight energy',
        'prediction_task': 'Round-trip energy after distance-based route planning',
        'route_policy': route_document['route_policy'], 'wind': 'synthetic uniform, steady; TO convention',
        'map_snapshot': map_meta, 'map_geojson_sha256': sha256_file(args.data_dir/'manhattan_buildings.geojson'),
        'dataset_sha256': sha256_file(csv_path), 'seed': args.seed, 'route_count': len(routes),
        'weather_case_count': len(cases), 'payloads_kg': payloads, 'speeds_mps': sorted(set(args.speeds)),
        'directions_to_degrees': sorted({d % 360 for d in args.directions}),
        'rows': len(rows), 'feasible_rows': sum(r['feasible'] for r in rows),
        'infeasible_rows': sum(not r['feasible'] for r in rows),
        'hub_cell': city.hub, 'cell_m': CELL_M, 'buffer_m': BUFFER_M,
        'flight_config': asdict(cfg), 'drone': asdict(Drone()), 'return_payload_kg': 0,
        'infeasible_energy': 'blank when there is no flyable route; never replaced with zero',
        'energy_scope': 'Cruise energy only; inherited mission time adds 2 minutes without handoff energy',
        'code_sha256': {f: sha256_file(ROOT/f) for f in [
            'day1_drone_mission.py', 'day3_astar_route.py', 'day4_wind_field.py',
            'day5_wind_energy_model.py', 'day8_manhattan.py', 'day9_generate_dataset.py']},
    }
    (args.output/'dataset_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    plot_routes(city, grid, routes, args.output/'routes_map.png')
    print(f"Saved {len(rows)} missions: {manifest['feasible_rows']} feasible, {manifest['infeasible_rows']} infeasible.")
    print('Output:', args.output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/day9')
    parser.add_argument('--destinations', type=int, default=24)
    parser.add_argument('--hub', type=int, nargs=2, default=DEFAULT_HUB)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--speeds', type=float, nargs='+', default=DEFAULT_SPEEDS)
    parser.add_argument('--directions', type=float, nargs='+', default=DEFAULT_DIRECTIONS)
    parser.add_argument('--payloads', type=float, nargs='+', default=DEFAULT_PAYLOADS)
    args = parser.parse_args()
    try:
        generate(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
