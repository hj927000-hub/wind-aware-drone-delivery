"""Distance A* versus directed wind-cost A*, outbound and return optimized separately."""
import argparse,csv,json,heapq
from pathlib import Path
from math import hypot
import numpy as np
from day2_city_grid import City
from day3_astar_route import astar,blocked_cells,neighbors
from day4_wind_field import WindConfig,build_wind_field
from day5_wind_energy_model import FlightConfig,edge,evaluate,mission_summary

def weighted_astar(city,field,start,goal,blocked,payload,cfg,use_heuristic=True):
    if not city.inside(start) or not city.inside(goal) or start in blocked or goal in blocked:return None
    wmax=float(np.nanmax(field.speed))
    base=1250+120*payload
    # Proven lower bound: power >= base, groundspeed <= airspeed + max wind.
    lower=(base/3600+cfg.time_weight)/(cfg.airspeed+wmax)
    def h(c):return hypot(c[0]-goal[0],c[1]-goal[1])*city.cell_m*lower if use_heuristic else 0
    frontier=[(h(start),0.,start)];best={start:0.};parent={}
    while frontier:
        _,g,cur=heapq.heappop(frontier)
        if g>best[cur]:continue
        if cur==goal:
            path=[cur]
            while cur in parent:cur=parent[cur];path.append(cur)
            return path[::-1]
        for nxt,_ in neighbors(city,cur,blocked):
            r=edge(city,field,cur,nxt,payload,cfg)
            if r is None:continue
            candidate=g+r['cost_wh']
            if candidate<best.get(nxt,float('inf')):
                best[nxt]=candidate;parent[nxt]=cur
                heapq.heappush(frontier,(candidate+h(nxt),candidate,nxt))
    return None

def run(speed=4,direction=0,buffer=1,risk_weight=.2,threshold=6,max_wind=10,time_weight=0,uniform=False):
    city=City();field=build_wind_field(city,WindConfig(speed,direction,threshold,not uniform))
    cfg=FlightConfig(soft_wind=threshold,max_wind=max_wind,risk_weight=risk_weight,time_weight=time_weight)
    blocked=blocked_cells(city,buffer)
    try:short=astar(city,city.hub,city.customer,blocked)
    except ValueError:short=None
    paths={'distance':(short,short[::-1] if short else None),'wind':(weighted_astar(city,field,city.hub,city.customer,blocked,1.5,cfg),weighted_astar(city,field,city.customer,city.hub,blocked,0,cfg))}
    results={};segments={}
    for name,(out,back) in paths.items():
        a=evaluate(city,field,out,1.5,cfg);b=evaluate(city,field,back,0,cfg)
        results[name]=mission_summary(a,b)
        segments[name]=[dict(leg=leg,**r) for leg,rows in [('outbound',a),('return',b)] for r in (rows or [])]
        if a is not None and b is not None:
            results[name]['high_wind_visits']=sum(int(field.risk[y,x]) for p in (out,back) for x,y in p)
    return city,field,cfg,paths,results,segments

def plot(city,field,paths,results,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    for ax,name in zip(axes,['distance','wind']):
        im=ax.imshow(field.speed,origin='lower',extent=(0,2000,0,2000),cmap='Blues',vmin=0,vmax=max(1,float(np.nanmax(field.speed))))
        for b in city.buildings:ax.add_patch(Rectangle((b.x*50,b.y*50),b.width*50,b.height*50,color='#475569'))
        for p,label,color,style in zip(paths[name],['Outbound','Return'],['#ef4444','#16a34a'],['-','--']):
            if p:
                coords=[city.cell_center_m(c) for c in p];x,y=zip(*coords);ax.plot(x,y,style,color=color,lw=2,label=label)
        for c,label in [(city.hub,'Hub'),(city.customer,'Customer')]:
            x,y=city.cell_center_m(c);ax.scatter(x,y,color='black',s=20);ax.annotate(label,(x,y),xytext=((-5,8) if label=='Customer' else (5,5)),ha=('right' if label=='Customer' else 'left'),textcoords='offset points')
        r=results[name];subtitle=f"{r['energy_wh']:.2f} Wh | {r['round_trip_m']:.0f} m" if 'energy_wh' in r else 'NO FLYABLE ROUND TRIP'
        ax.set(title=f'{name.upper()} route\n{subtitle}',xlabel='x [m]',ylabel='y [m]',aspect='equal')
        if ax.get_legend_handles_labels()[0]:ax.legend(loc='upper left')
    fig.colorbar(im,ax=axes,label='Synthetic wind [m/s]',shrink=.8)
    fig.suptitle('DAY 6 | Same wind, same vehicle, different route objective')
    fig.savefig(dest,dpi=150);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in [('speed',4),('direction',0),('risk-weight',.2),('threshold',6),('max-wind',10),('time-weight',0)]:p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--buffer',type=int,default=1);p.add_argument('--uniform',action='store_true');p.add_argument('--output',default='outputs/day6')
    a=p.parse_args();kwargs=vars(a).copy();out=Path(kwargs.pop('output'));out.mkdir(parents=True,exist_ok=True)
    try:city,field,cfg,paths,results,segments=run(**kwargs)
    except ValueError as e:p.error(str(e))
    (out/'results.json').write_text(json.dumps(dict(settings=kwargs,results=results),indent=2))
    keys=list(dict.fromkeys(k for r in results.values() for k in r))
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['route']+keys);w.writeheader();w.writerows(dict(route=k,**r) for k,r in results.items())
    for name,rows in segments.items():
        with (out/(name+'_segments.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['leg','sequence']);w.writeheader();w.writerows(rows)
    (out/'paths.json').write_text(json.dumps(paths,indent=2))
    plot(city,field,paths,results,out/'comparison.png')
    print(json.dumps(results,indent=2));print('Outputs:',out)
if __name__=='__main__':main()
