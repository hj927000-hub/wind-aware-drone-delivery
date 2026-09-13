# Wind-Aware Drone Delivery Simulator

A Python learning project exploring how buildings, wind, and routing objectives affect drone delivery distance, flight time, and battery consumption.

The main question is: **Does the shortest route also use the least energy?**

## Current scope

The simulator represents a synthetic city as a 2D grid. It compares a distance-minimizing A* route with a route that minimizes estimated energy plus optional time and wind-exposure penalties.

| Stage | Implemented functionality |
| --- | --- |
| Day 1 | Mission energy, time, state of charge, and battery usage estimates |
| Day 2 | Grid-based city, building footprints, hub, and delivery location |
| Day 3 | A* routing with building buffers and diagonal corner checks |
| Day 4 | Synthetic wind vectors and high-wind regions |
| Day 5 | Wind-dependent ground speed, travel time, and segment energy |
| Day 6 | Wind-aware A*, separate outbound and return planning, and route comparison |

Day 7 focuses on documenting and sharing the first week's implementation and experiments.

## Quick start

Use Python 3.10 or newer. Keep the Day 1–6 Python files in the same directory.

```bash
python3 -m pip install numpy matplotlib
python3 day5_wind_energy_model.py
python3 day6_wind_aware_routing.py
```

The Day 6 command writes results to `outputs/day6/`:

- `comparison.png`: distance-based and wind-aware routes
- `results.json` and `comparison.csv`: mission metrics
- `paths.json`: ordered route coordinates
- `distance_segments.csv` and `wind_segments.csv`: segment-level calculations

In the plot, red is the outbound route and dashed green is the return route. Both planning methods are evaluated in the same wind field.

## Experiments

Run these commands from the project directory. Keep the city and vehicle settings unchanged when comparing scenarios.

```bash
# No wind: compare the two planning methods
python3 day6_wind_aware_routing.py --speed 0 --risk-weight 0 --output outputs/calm

# Wind: minimize estimated energy
python3 day6_wind_aware_routing.py --speed 6 --risk-weight 0 --output outputs/energy_only

# Same wind: increase the wind-exposure penalty
python3 day6_wind_aware_routing.py --speed 6 --risk-weight 50 --output outputs/high_risk_weight

# Uniform wind above the configured limit: no flyable route
python3 day6_wind_aware_routing.py --speed 13 --uniform --output outputs/blocked
```

### Reference results

These are simulation results from the default bundled city and vehicle settings. Modified inputs may produce different results. Distances and energies below are for the round trip.

| Scenario | Planner | Distance (m) | Energy (Wh) | Wind-exposure score (risk-weighted seconds) |
| --- | --- | ---: | ---: | ---: |
| No wind, risk weight 0 | Distance / wind-aware | 5,179.90 | 160.67 | 0 |
| Base wind 6 m/s, risk weight 0 | Distance | 5,179.90 | 196.05 | 0.7937 |
| Base wind 6 m/s, risk weight 0 | Wind-aware | 5,179.90 | 193.40 | 0.3772 |
| Base wind 6 m/s, risk weight 50 | Wind-aware | 5,209.19 | 194.55 | 0.1467 |

In this example, wind-aware routing uses less energy without reducing distance. Increasing the risk weight then selects a longer route that uses slightly more energy but has a lower modeled wind-exposure score.

## How routing works

For each neighboring cell, the simulator estimates ground speed from the desired travel direction and wind vector. It then calculates travel time and energy:

```text
time_s = distance_m / ground_speed_mps
energy_wh = power_w * time_s / 3600

risk_s = time_s * max(0, wind_speed / soft_wind - 1)^2
cost_wh = energy_wh + time_weight * time_s + risk_weight * risk_s
```

`energy_wh` is estimated battery energy consumption. `cost_wh` is a selection score expressed in Wh-equivalent units; its penalties are not additional battery consumption.

- `g`: accumulated selection cost to a candidate cell
- `h`: a lower-bound estimate of the remaining selection cost
- `f = g + h`: priority for inspecting a candidate
- `best`: the lowest discovered `g` for each cell
- `parent`: the predecessor used to reconstruct a route
- `frontier`: a priority queue of candidates

A* updates `best` and `parent` when it discovers a cheaper way to reach a cell. It inspects candidates in increasing `f` order and reconstructs the route when the goal is popped.

The distance baseline returns along its outbound path in reverse. The wind-aware planner optimizes outbound and return paths separately because wind direction relative to travel and payload differ between legs.

## Tests

```bash
python3 -m unittest test_day56.py
```

The tests cover analytical wind cases, prohibited segments, agreement with the no-wind model, A* cost agreement with Dijkstra, infeasible missions, risk-weight trade-offs, directional asymmetry, and invalid inputs.

## What I learned

- Equal route lengths do not imply identical paths or energy consumption.
- Segment energy is estimated before route selection; total mission energy is calculated from the selected segments.
- Cost weights change route preferences, not the physical energy required for an unchanged route.
- At constant power and zero risk weight, energy and time are proportional, so adding a time penalty alone does not change which routes are optimal.
- A lower modeled risk score is not evidence of a lower real-world accident probability.

## Assumptions and limitations

- The map and wind field are synthetic. Real geographic and weather data are not yet integrated.
- The model uses a fixed flight layer, constant airspeed, and payload-dependent power. The extra crosswind power coefficient is zero by default.
- Segment wind is approximated using the average of its endpoint wind vectors. This can hide short high-wind exposure, including producing zero risk penalty for a segment adjacent to a high-wind cell.
- Wind thresholds and risk weights are educational settings, not validated aircraft operating limits.
- Gusts, detailed aircraft dynamics, turn energy, takeoff, landing, and charging losses are not modeled. A fixed two-minute handoff adds time but no energy.
- Landing SoC is checked after route planning; battery reserve is not a constraint within the A* search state.

This project is an educational simulator, not flight-control software or a validated operational planning tool.

## Next steps

- Replace the synthetic city with a small area of San Francisco or New York using real building footprints.
- Convert geographic coordinates to a local metric grid and reuse the routing pipeline.
- Improve the wind-exposure approximation and compare its effect on route selection.
- Explore a regression model trained on simulation outputs, with a clear distinction between learning the simulator and predicting real flight performance.
