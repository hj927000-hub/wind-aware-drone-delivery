"""Day 1 — baseline drone delivery mission simulator.

One drone flies from a hub to one customer and returns in zero wind.
All numbers below are editable assumptions, not validated aircraft data.
The purpose of Day 1 is to establish a transparent calculation pipeline:

    mission -> flight time -> energy -> SoC -> EFC -> battery/electricity cost

Run:
    python3 day1_drone_mission.py
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Drone:
    cruise_speed_mps: float = 12.0
    cruise_power_empty_w: float = 1_250.0
    payload_power_per_kg_w: float = 120.0
    battery_capacity_wh: float = 1_500.0
    usable_battery_fraction: float = 0.80
    initial_soc_pct: float = 90.0
    minimum_landing_soc_pct: float = 20.0


@dataclass(frozen=True)
class Economics:
    electricity_price_per_kwh: float = 0.18
    battery_pack_price_usd: float = 750.0
    expected_pack_life_efc: float = 600.0


@dataclass(frozen=True)
class Mission:
    one_way_distance_km: float = 5.0
    payload_outbound_kg: float = 1.5
    payload_return_kg: float = 0.0
    ground_time_min: float = 2.0


@dataclass(frozen=True)
class LegResult:
    name: str
    distance_km: float
    payload_kg: float
    time_min: float
    energy_wh: float


def simulate_leg(
    name: str,
    distance_km: float,
    payload_kg: float,
    drone: Drone,
) -> LegResult:
    """Simulate one constant-speed, zero-wind cruise segment."""
    distance_m = distance_km * 1_000.0
    time_seconds = distance_m / drone.cruise_speed_mps
    power_w = (
        drone.cruise_power_empty_w
        + payload_kg * drone.payload_power_per_kg_w
    )
    energy_wh = power_w * time_seconds / 3_600.0

    return LegResult(
        name=name,
        distance_km=distance_km,
        payload_kg=payload_kg,
        time_min=time_seconds / 60.0,
        energy_wh=energy_wh,
    )


def simulate_mission(
    drone: Drone,
    mission: Mission,
    economics: Economics,
) -> dict[str, float | bool | list[LegResult]]:
    outbound = simulate_leg(
        "Hub -> Customer",
        mission.one_way_distance_km,
        mission.payload_outbound_kg,
        drone,
    )
    inbound = simulate_leg(
        "Customer -> Hub",
        mission.one_way_distance_km,
        mission.payload_return_kg,
        drone,
    )
    legs = [outbound, inbound]

    total_energy_wh = sum(leg.energy_wh for leg in legs)
    flight_time_min = sum(leg.time_min for leg in legs)
    total_time_min = flight_time_min + mission.ground_time_min

    # SoC is referenced to nominal pack capacity.
    soc_drop_pct = total_energy_wh / drone.battery_capacity_wh * 100.0
    final_soc_pct = drone.initial_soc_pct - soc_drop_pct

    # EFC measures throughput relative to the usable energy window.
    usable_pack_energy_wh = (
        drone.battery_capacity_wh * drone.usable_battery_fraction
    )
    efc_used = total_energy_wh / usable_pack_energy_wh

    electricity_cost_usd = (
        total_energy_wh / 1_000.0
        * economics.electricity_price_per_kwh
    )
    battery_depreciation_usd = (
        economics.battery_pack_price_usd
        * efc_used
        / economics.expected_pack_life_efc
    )

    return {
        "legs": legs,
        "total_distance_km": 2.0 * mission.one_way_distance_km,
        "flight_time_min": flight_time_min,
        "total_time_min": total_time_min,
        "total_energy_wh": total_energy_wh,
        "soc_drop_pct": soc_drop_pct,
        "final_soc_pct": final_soc_pct,
        "efc_used": efc_used,
        "electricity_cost_usd": electricity_cost_usd,
        "battery_depreciation_usd": battery_depreciation_usd,
        "energy_and_battery_cost_usd": (
            electricity_cost_usd + battery_depreciation_usd
        ),
        "mission_feasible": final_soc_pct >= drone.minimum_landing_soc_pct,
    }


def print_report(result: dict[str, float | bool | list[LegResult]]) -> None:
    print("\nDAY 1 — BASELINE DRONE DELIVERY (ZERO WIND)")
    print("=" * 52)

    legs = result["legs"]
    assert isinstance(legs, list)
    for leg in legs:
        print(
            f"{leg.name:<20} | {leg.time_min:>5.2f} min | "
            f"{leg.energy_wh:>7.1f} Wh | payload {leg.payload_kg:.1f} kg"
        )

    print("-" * 52)
    print(f"Round-trip distance       : {result['total_distance_km']:.2f} km")
    print(f"Total mission time        : {result['total_time_min']:.2f} min")
    print(f"Flight energy             : {result['total_energy_wh']:.1f} Wh")
    print(f"Battery SoC               : 90.0% -> {result['final_soc_pct']:.1f}%")
    print(f"Equivalent full cycles    : {result['efc_used']:.4f} EFC")
    print(f"Electricity cost          : ${result['electricity_cost_usd']:.3f}")
    print(f"Battery depreciation      : ${result['battery_depreciation_usd']:.3f}")
    print(f"Energy + battery cost     : ${result['energy_and_battery_cost_usd']:.3f}")
    print(f"Landing reserve satisfied : {result['mission_feasible']}")


if __name__ == "__main__":
    simulation_result = simulate_mission(
        drone=Drone(),
        mission=Mission(),
        economics=Economics(),
    )
    print_report(simulation_result)
