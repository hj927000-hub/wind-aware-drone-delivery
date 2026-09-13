"""Day 3: shortest path on the Day 2 grid, then reuse Day 1 energy.

python3 day3_astar_route.py --buffer-cells 1
Synthetic fixed-altitude model, not flight guidance. Costs are distance [m].
"""
import argparse
import csv
import heapq
from math import hypot
from pathlib import Path

from day2_city_grid import City
from day1_drone_mission import Drone, Economics, Mission, simulate_mission


def blocked_cells(city, buffer_cells=1):
    """Square/Chebyshev dilation: block N extra cells around footprints."""
    if type(buffer_cells) is not int or buffer_cells < 0:
        raise ValueError('buffer_cells must be a non-negative integer.')
    blocked = set()
    for b in city.buildings:
        for y in range(max(0, b.y-buffer_cells),
                       min(city.rows, b.y+b.height+buffer_cells)):
            for x in range(max(0, b.x-buffer_cells),
                           min(city.columns, b.x+b.width+buffer_cells)):
                blocked.add((x, y))
    return blocked


def neighbors(city, cell, blocked):
    """8 directions; diagonals cannot cut a blocked corner."""
    x, y = cell
    for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1),
                   (1, 1), (-1, 1), (-1, -1), (1, -1)):
        nxt = (x+dx, y+dy)
        if not city.inside(nxt) or nxt in blocked:
            continue
        if dx and dy and ((x+dx, y) in blocked or (x, y+dy) in blocked):
            continue
        yield nxt, hypot(dx, dy) * city.cell_m


def heuristic(city, a, b):
    """Straight distance in meters: a lower bound on remaining path cost."""
    return hypot(a[0]-b[0], a[1]-b[1]) * city.cell_m


def astar(city, start, goal, blocked):
    if not city.inside(start) or not city.inside(goal):
        raise ValueError('Endpoint outside map.')
    if start in blocked or goal in blocked:
        raise ValueError('Hub or customer is inside a building/buffer. Move it or reduce buffer.')
    # g = distance traveled so far; f = g + estimated remaining distance.
    frontier = [(heuristic(city, start, goal), 0.0, start)]
    distance = {start: 0.0}
    parent = {}
    while frontier:
        _, g, current = heapq.heappop(frontier)
        if g > distance[current]:
            continue  # An older, longer candidate for this cell.
        if current == goal:
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            return list(reversed(path))
        for nxt, step_m in neighbors(city, current, blocked):
            candidate = g + step_m
            if candidate < distance.get(nxt, float('inf')):
                distance[nxt] = candidate
                parent[nxt] = current
                heapq.heappush(frontier, (candidate + heuristic(city, nxt, goal),
                                          candidate, nxt))
    return None  # No path; do not report a zero-energy successful mission.


def path_distance_m(city, path):
    return sum(heuristic(city, a, b) for a, b in zip(path, path[1:]))


def compare_energy(city, distance_m):
    """Same map endpoints, not Day 1's original 5 km mission."""
    rows = []
    for label, length in [('Straight reference (ignores buildings)', city.straight_distance_m()),
                          ('A* route', distance_m)]:
        r = simulate_mission(Drone(), Mission(one_way_distance_km=length/1000), Economics())
        rows.append(dict(scenario=label, one_way_m=length,
                         round_trip_m=2*length, mission_min=r['total_time_min'],
                         energy_wh=r['total_energy_wh'], soc_pct=r['final_soc_pct'],
                         efc_usable_window=r['efc_used'],
                         electricity_usd=r['electricity_cost_usd'],
                         battery_wear_usd=r['battery_depreciation_usd'],
                         energy_battery_usd=r['energy_and_battery_cost_usd'],
                         landing_reserve_met=r['mission_feasible']))
    return rows


def save_plot(city, blocked, path, buffer_cells, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    grid = [[2 if not city.is_free((x, y)) else int((x, y) in blocked)
             for x in range(city.columns)] for y in range(city.rows)]
    fig, ax = plt.subplots(figsize=(9, 8), layout='constrained')
    ax.imshow(grid, origin='lower', interpolation='nearest', vmin=0, vmax=2,
              extent=(0, city.columns*city.cell_m, 0, city.rows*city.cell_m),
              cmap=ListedColormap(['#f1f5f9', '#fcd9a5', '#475569']))
    a, b = city.cell_center_m(city.hub), city.cell_center_m(city.customer)
    ax.plot([a[0], b[0]], [a[1], b[1]], '--', color='#dc2626', lw=1.5,
            label='Straight reference (ignores buildings)')
    coords = [city.cell_center_m(c) for c in path]
    ax.plot(*zip(*coords), color='#059669', lw=2.5, label='A* outbound; return retraces route')
    ax.scatter(*a, s=110, marker='s', color='#0284c7', zorder=5, label='Hub')
    ax.scatter(*b, s=180, marker='*', color='#e17020', zorder=5, label='Customer')
    handles, _ = ax.get_legend_handles_labels()
    handles += [Patch(color='#475569', label='Buildings'),
                Patch(color='#fcd9a5', label=f'Buffer: {buffer_cells} cells')]
    fig.legend(handles=handles, loc='outside lower center', ncols=2, fontsize=9)
    ax.set(title=f'DAY 3 | A* obstacle avoidance | buffer {buffer_cells} cells\n'
                 f'Straight {city.straight_distance_m():.1f} m | Route {path_distance_m(city, path):.1f} m',
           xlabel='Local x [m]', ylabel='Local y [m]')
    ax.grid(alpha=0.18)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--buffer-cells', type=int, default=1)
    args = parser.parse_args()
    try:
        city = City()  # Uses your day2_city_grid.py settings.
        blocked = blocked_cells(city, args.buffer_cells)
        path = astar(city, city.hub, city.customer, blocked)
    except ValueError as exc:
        parser.exit(2, f'Input error: {exc}\n')
    if path is None:
        parser.exit(2, 'No route: free space is disconnected. No new output generated.\n')
    output = Path(__file__).resolve().parent / 'outputs'
    output.mkdir(exist_ok=True)
    stem = f'day3_buffer_{args.buffer_cells}'
    rows = compare_energy(city, path_distance_m(city, path))
    with (output / f'{stem}_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / f'{stem}_path.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['leg', 'sequence', 'x_cell', 'y_cell', 'x_m', 'y_m'])
        for leg, route in [('outbound', path), ('return', list(reversed(path)))]:
            for i, cell in enumerate(route):
                writer.writerow([leg, i, *cell, *city.cell_center_m(cell)])
    save_plot(city, blocked, path, args.buffer_cells, output / f'{stem}_route.png')
    print('Synthetic grid; no wind, turning energy, takeoff or landing energy.')
    print('Return retraces outbound path. Day 1 energy and usable-window EFC assumptions retained.')
    for r in rows:
        print(f"{r['scenario']}: {r['one_way_m']:.2f} m one way | "
              f"{r['mission_min']:.2f} min round trip incl. 2 min handoff | "
              f"{r['energy_wh']:.2f} Wh | SoC {r['soc_pct']:.2f}% | "
              f"energy+battery ${r['energy_battery_usd']:.3f}")
    print(f'Outputs: {output / stem}*')


if __name__ == '__main__':
    main()
