"""Fixed airspeed, track-holding, steady synthetic wind. Educational model."""
from dataclasses import dataclass
from math import hypot,sqrt,isfinite
import csv
from pathlib import Path
from day1_drone_mission import Drone,Economics

@dataclass(frozen=True)
class FlightConfig:
    airspeed: float = 12.0
    soft_wind: float = 6.0
    max_wind: float = 10.0
    cross_power_coeff: float = 0.0
    time_weight: float = 0.0  # Wh / second
    risk_weight: float = 0.2  # Wh / risk-second
    def __post_init__(self):
        vals=vars(self)
        if not all(isfinite(v) for v in vals.values()):raise ValueError('Inputs must be finite.')
        if self.airspeed<=0 or self.soft_wind<=0 or self.max_wind<self.soft_wind or min(self.cross_power_coeff,self.time_weight,self.risk_weight)<0:raise ValueError('Invalid flight settings.')

def segment(distance_m,dx,dy,u,v,payload=1.5,cfg=FlightConfig()):
    """Return None if wind/track conditions prohibit the segment.
    w_parallel = wind dot desired-track unit vector.
    Track held by cancelling lateral wind with opposite air-velocity component.
    """
    if not all(isfinite(x) for x in [distance_m,dx,dy,u,v,payload]) or distance_m<=0 or payload<0 or hypot(dx,dy)==0:raise ValueError('Invalid segment.')
    norm=hypot(dx,dy);tx,ty=dx/norm,dy/norm
    parallel=u*tx+v*ty;cross=-u*ty+v*tx;wind=hypot(u,v)
    if wind>cfg.max_wind or abs(cross)>=cfg.airspeed:return None
    ground=sqrt(cfg.airspeed**2-cross**2)+parallel
    if ground<=1e-9:return None
    seconds=distance_m/ground
    base=Drone().cruise_power_empty_w+payload*Drone().payload_power_per_kg_w
    power=base*(1+cfg.cross_power_coeff*(cross/cfg.airspeed)**2)
    energy=power*seconds/3600
    risk=seconds*max(0,wind/cfg.soft_wind-1)**2
    return dict(distance_m=distance_m,parallel_mps=parallel,cross_mps=cross,ground_mps=ground,time_s=seconds,energy_wh=energy,risk_s=risk,cost_wh=energy+cfg.time_weight*seconds+cfg.risk_weight*risk)

def edge(city,field,a,b,payload,cfg):
    if not city.is_free(a) or not city.is_free(b):return None
    x,y=a;xx,yy=b
    if max(field.speed[y,x],field.speed[yy,xx])>cfg.max_wind:return None
    u=float((field.u[y,x]+field.u[yy,xx])/2);v=float((field.v[y,x]+field.v[yy,xx])/2)
    return segment(hypot(xx-x,yy-y)*city.cell_m,xx-x,yy-y,u,v,payload,cfg)

def evaluate(city,field,path,payload,cfg):
    if path is None:return None
    rows=[]
    for i,(a,b) in enumerate(zip(path,path[1:])):
        r=edge(city,field,a,b,payload,cfg)
        if r is None:return None
        rows.append(dict(sequence=i,ax=a[0],ay=a[1],bx=b[0],by=b[1],**r))
    return rows

def mission_summary(outbound,inbound):
    if outbound is None or inbound is None:return dict(feasible=False,reason='No flyable complete round trip')
    rows=outbound+inbound;d=Drone();eco=Economics()
    energy=sum(r['energy_wh'] for r in rows);soc=d.initial_soc_pct-energy/d.battery_capacity_wh*100
    efc=energy/(d.battery_capacity_wh*d.usable_battery_fraction)
    return dict(feasible=soc>=d.minimum_landing_soc_pct,reason='OK' if soc>=d.minimum_landing_soc_pct else 'Landing reserve not met',round_trip_m=sum(r['distance_m'] for r in rows),mission_min=sum(r['time_s'] for r in rows)/60+2,energy_wh=energy,final_soc_pct=soc,efc=efc,electricity_usd=energy/1000*eco.electricity_price_per_kwh,battery_wear_usd=efc*eco.battery_pack_price_usd/eco.expected_pack_life_efc,risk_s=sum(r['risk_s'] for r in rows),cost_wh=sum(r['cost_wh'] for r in rows))

def main():
    rows=[]
    for label,u,v in [('calm',0,0),('tailwind',4,0),('headwind',-4,0),('crosswind',0,4),('blocked wind',13,0)]:
        r=segment(1000,1,0,u,v)
        rows.append(dict(scenario=label,feasible=r is not None,**(r or {})))
        print(label, 'NOT FLYABLE' if r is None else f"{r['ground_mps']:.2f} m/s | {r['time_s']:.2f} s | {r['energy_wh']:.2f} Wh")
    out=Path('outputs');out.mkdir(exist_ok=True)
    with (out/'day5_examples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
