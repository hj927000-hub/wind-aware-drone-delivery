"""Real Manhattan footprints -> metric grid -> existing 2D A*.

Height tags are preserved for a 2.5D preview; altitude is NOT a search variable.
Default run uses a bundled OSM snapshot and requires no network access.
"""
import argparse
from collections import deque
from dataclasses import dataclass
import json
from math import hypot, isfinite
from pathlib import Path
import re

import numpy as np
from pyproj import Transformer
import shapely
from shapely.affinity import translate
from shapely.geometry import box, mapping, shape, LineString
from shapely.ops import transform, triangulate, unary_union

from day3_astar_route import astar, neighbors
from day4_wind_field import WindConfig, build_wind_field
from day5_wind_energy_model import FlightConfig, evaluate, mission_summary
from day6_wind_aware_routing import weighted_astar

ROOT = Path(__file__).resolve().parent


def parse_height_m(value):
    """Parse an explicit OSM height; never estimate height from floor count."""
    if value is None:
        return None
    match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(m|ft|feet|\')?\s*', str(value))
    if not match:
        return None
    height = float(match[1]) * (0.3048 if match[2] in ('ft', 'feet', "'") else 1)
    return height if isfinite(height) and height > 0 else None


def load_buildings(data_dir):
    """Geographic lon/lat -> UTM meters -> local (0..1000 m) coordinates."""
    meta = json.loads((data_dir / 'manhattan_metadata.json').read_text())
    raw = json.loads((data_dir / 'manhattan_buildings.geojson').read_text())
    forward = Transformer.from_crs(4326, meta['metric_crs'], always_xy=True)
    ox, oy = meta['origin_utm_m']; size = meta['side_m']
    extent = box(0, 0, size, size)
    buildings = []
    for feature in raw['features']:
        geom = shapely.make_valid(shape(feature['geometry']))
        geom = translate(transform(forward.transform, geom), -ox, -oy)
        geom = geom.intersection(extent)
        parts = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
        parts = [g for g in parts if g.geom_type == 'Polygon' and g.area > 0]
        if not parts:
            continue
        props = dict(feature['properties'])
        props['height_m'] = parse_height_m(props.get('height'))
        props['height_status'] = 'osm_tag_unverified' if props['height_m'] is not None else 'unknown'
        buildings.append((unary_union(parts), props))
    if not buildings:
        raise ValueError('No footprints in the selected area.')
    return buildings, meta


def rasterize(geometry, side_m, cell_m, buffer_m):
    """Block every cell touching a footprint expanded by buffer_m meters."""
    if not all(isfinite(v) for v in (side_m, cell_m, buffer_m)):
        raise ValueError('Grid settings must be finite.')
    if side_m <= 0 or cell_m <= 0 or buffer_m < 0:
        raise ValueError('Use positive dimensions and a nonnegative buffer.')
    n = round(side_m / cell_m)
    if n < 1 or n > 1000 or not np.isclose(n * cell_m, side_m):
        raise ValueError('Cell size must divide the map width; at most 1000 cells per side.')
    yy, xx = np.mgrid[0:n, 0:n]
    cells = shapely.box(xx*cell_m, yy*cell_m, (xx+1)*cell_m, (yy+1)*cell_m)
    obstacle = geometry.buffer(buffer_m) if buffer_m else geometry
    # Intersections include boundary touches. This is intentionally conservative.
    return np.asarray(shapely.intersects(cells, obstacle), dtype=bool)


@dataclass
class RealCity:
    raw_grid: np.ndarray
    cell_m: float
    hub: tuple = (0, 0)
    customer: tuple = (0, 0)
    buildings: tuple = ()  # Day 4 local building effects disabled; uniform wind only.

    @property
    def rows(self): return self.raw_grid.shape[0]

    @property
    def columns(self): return self.raw_grid.shape[1]

    def inside(self, cell):
        x, y = cell
        return isinstance(x, (int, np.integer)) and isinstance(y, (int, np.integer)) and 0 <= x < self.columns and 0 <= y < self.rows

    def is_free(self, cell):
        return self.inside(cell) and not self.raw_grid[cell[1], cell[0]]

    def occupancy(self): return self.raw_grid.astype(int).tolist()

    def cell_center_m(self, cell):
        if not self.inside(cell): raise ValueError('Cell is outside the map.')
        return ((cell[0]+.5)*self.cell_m, (cell[1]+.5)*self.cell_m)


def largest_component(city, blocked):
    """Connectivity uses Day 3's same diagonal corner rule."""
    unseen = {(x, y) for y in range(city.rows) for x in range(city.columns)
              if (x, y) not in blocked}
    largest = set()
    while unseen:
        start = min(unseen); unseen.remove(start)
        component = {start}; queue = deque([start])
        while queue:
            for nxt, _ in neighbors(city, queue.popleft(), blocked):
                if nxt in unseen:
                    unseen.remove(nxt); component.add(nxt); queue.append(nxt)
        if len(component) > len(largest): largest = component
    return largest


def demo_endpoints(city, component):
    """Pick reproducible illustrative endpoints in the largest component."""
    if len(component) < 2: raise ValueError('Fewer than two connected free cells.')
    targets = [(city.columns*.15, city.rows*.15), (city.columns*.85, city.rows*.85)]
    selected = [min(component, key=lambda c: ((c[0]-x)**2+(c[1]-y)**2, c)) for x,y in targets]
    if selected[0] == selected[1]:
        selected[1] = max(component, key=lambda c: (hypot(c[0]-selected[0][0],c[1]-selected[0][1]),c))
    return selected


def route_mission(city, field, blocked, cfg):
    short = astar(city, city.hub, city.customer, blocked)
    wind_out = weighted_astar(city, field, city.hub, city.customer, blocked, 1.5, cfg)
    wind_back = weighted_astar(city, field, city.customer, city.hub, blocked, 0, cfg)
    paths = {'distance': (short, short[::-1] if short else None), 'wind': (wind_out, wind_back)}
    results = {}
    for key, (out, back) in paths.items():
        results[key] = mission_summary(evaluate(city, field, out, 1.5, cfg),
                                       evaluate(city, field, back, 0, cfg))
    return paths, results


def polygon_parts(geom):
    return list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]


def plot_map(buildings, city, buffered, paths, results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), layout='constrained')
    side = city.columns*city.cell_m
    for geom, _ in buildings:
        for poly in polygon_parts(geom):
            poly = shapely.geometry.polygon.orient(poly)
            vertices=[]; codes=[]
            for ring in [poly.exterior, *poly.interiors]:
                coords=list(ring.coords); vertices.extend(coords)
                codes.extend([MplPath.MOVETO]+[MplPath.LINETO]*(len(coords)-2)+[MplPath.CLOSEPOLY])
            axes[0].add_patch(PathPatch(MplPath(vertices, codes), facecolor='#40526a',edgecolor='#26384f',lw=.2))
    category = np.where(city.raw_grid, 2, np.where(buffered, 1, 0))
    axes[1].imshow(category, origin='lower', extent=(0, side, 0, side),
                   cmap=ListedColormap(['#f5f8fb','#f7d8a8','#40526a']),vmin=0,vmax=2,interpolation='nearest')
    for ax in axes:
        for key,color,style,label in [('distance','#ea6b31','--','Shortest distance (out)'),('wind','#078275','-','Wind-aware (out)')]:
            path=paths[key][0]
            if path:
                x,y=zip(*(city.cell_center_m(c) for c in path))
                ax.plot(x,y,style,color=color,lw=2,label=label,zorder=5)
        for c,name,marker,color in [(city.hub,'Hub','s','#2473b7'),(city.customer,'Customer','*','#b3327e')]:
            x,y=city.cell_center_m(c);ax.scatter(x,y,s=80,marker=marker,color=color,zorder=6)
            ax.annotate(f'{name} {c}',(x,y),xytext=(5,8),textcoords='offset points',fontsize=8)
        ax.set(xlim=(0,side),ylim=(0,side),aspect='equal',xlabel='Local east [m]',ylabel='Local north [m]')
        if ax.get_legend_handles_labels()[0]:ax.legend(loc='upper left',fontsize=8)
    axes[0].set_title('Actual OSM building footprints')
    axes[1].set_title(f'Planning grid: {city.columns} x {city.rows} | cell {city.cell_m:g} m\nOrange = additional buffer cells')
    fig.suptitle('DAY 8 | Manhattan / Times Square | 2D route planning',fontsize=15)
    r=results['wind']
    stats=f"Round trip {r['round_trip_m']:.1f} m | {r['energy_wh']:.2f} Wh | SoC {r['final_soc_pct']:.2f}%" if 'energy_wh' in r else 'No complete flyable route'
    fig.supxlabel(stats+'\n© OpenStreetMap contributors · synthetic uniform wind · educational model',fontsize=9)
    fig.savefig(output,dpi=150);plt.close(fig)


def plot_height_preview(buildings, side, output):
    """Extrude explicit height tags only. Unknown heights stay as ground outlines."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(11,8),layout='constrained');ax=fig.add_subplot(projection='3d')
    height_max=1
    for geom,props in buildings:
        height=props['height_m']
        for poly in polygon_parts(geom):
            if height is None:
                x,y=poly.exterior.xy;ax.plot(x,y,np.zeros(len(x)),color='#adb5bf',lw=.45)
                continue
            height_max=max(height_max,height);faces=[]
            for ring in [poly.exterior,*poly.interiors]:
                c=list(ring.coords)
                for a,b in zip(c,c[1:]):faces.append([(a[0],a[1],0),(b[0],b[1],0),(b[0],b[1],height),(a[0],a[1],height)])
            for tri in triangulate(poly):
                if poly.covers(tri):faces.append([(x,y,height) for x,y in list(tri.exterior.coords)[:-1]])
            ax.add_collection3d(Poly3DCollection(faces,facecolors='#5c8cb3',edgecolors='#3e6484',linewidths=.1,alpha=.75))
    ax.set(xlim=(0,side),ylim=(0,side),zlim=(0,height_max*1.05),xlabel='Local east [m]',ylabel='Local north [m]',zlabel='Tagged height [m]')
    ax.set_box_aspect((side,side,height_max));ax.view_init(elev=35,azim=-65)
    ax.set_title('2.5D building preview — not a 3D flight route\nFlat ground; OSM height tags are not independently verified')
    ax.legend(handles=[Patch(facecolor='#5c8cb3',label='Explicit OSM height'),Patch(facecolor='#adb5bf',label='Unknown height: ground outline')],loc='upper left')
    fig.supxlabel('© OpenStreetMap contributors · missing heights are not estimated · no altitude planning',fontsize=9)
    fig.savefig(output,dpi=150);plt.close(fig)


def run(args):
    buildings,meta=load_buildings(args.data_dir)
    geometry=unary_union([g for g,_ in buildings]);side=meta['side_m']
    raw=rasterize(geometry,side,args.cell_m,0)
    buffered=rasterize(geometry,side,args.cell_m,args.buffer_m)
    city=RealCity(raw,args.cell_m)
    blocked={(int(x),int(y)) for y,x in np.argwhere(buffered)}
    component=largest_component(city,blocked)
    if (args.hub is None)!=(args.customer is None):raise ValueError('Provide both --hub and --customer, or neither.')
    endpoints=demo_endpoints(city,component) if args.hub is None else [tuple(args.hub),tuple(args.customer)]
    city.hub,city.customer=endpoints
    for c in endpoints:
        if not city.inside(c) or c in blocked:raise ValueError(f'Endpoint {c} is outside the map or in a building/buffer. No automatic relocation.')
    wind=WindConfig(args.speed,args.direction,args.threshold,False)
    field=build_wind_field(city,wind)
    cfg=FlightConfig(soft_wind=args.threshold,max_wind=args.max_wind,risk_weight=args.risk_weight)
    paths,results=route_mission(city,field,blocked,cfg)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'grid.npz',occupied=raw,blocked=buffered,cell_m=args.cell_m,buffer_m=args.buffer_m)
    height_count=sum(p['height_m'] is not None for _,p in buildings)
    report={'map':{**meta,'buildings_in_roi':len(buildings),'explicit_height_count':height_count,
                   'unknown_height_count':len(buildings)-height_count,'rows':city.rows,'columns':city.columns,
                   'cell_m':args.cell_m,'buffer_m':args.buffer_m,'blocked_cells':len(blocked),
                   'largest_connected_free_component':len(component),'hub':city.hub,'customer':city.customer,
                   'endpoint_selection':'automatic_demo_largest_component' if args.hub is None else 'user_specified'},
            'wind':{'synthetic':True,'uniform':True,'speed_mps':args.speed,'direction_to_degrees':args.direction,
                    'soft_wind_mps':args.threshold,'max_wind_mps':args.max_wind,'risk_weight':args.risk_weight},
            'routing_dimensions':2,'results':results}
    (args.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (args.output/'paths.json').write_text(json.dumps(paths))
    features=[{'type':'Feature','properties':p,'geometry':mapping(g)} for g,p in buildings]
    (args.output/'buildings_local_m.json').write_text(json.dumps({'coordinate_system':'local_meters_not_lon_lat','origin_utm_m':meta['origin_utm_m'],'metric_crs':meta['metric_crs'],'features':features},allow_nan=False))
    plot_map(buildings,city,buffered,paths,results,args.output/'day8_map.png')
    if args.preview_3d:plot_height_preview(buildings,side,args.output/'day8_height_preview.png')
    print(json.dumps(report,indent=2))
    print('Outputs:',args.output)
    return city,buildings,buffered,paths,results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/day8')
    parser.add_argument('--cell-m',type=float,default=5)
    parser.add_argument('--buffer-m',type=float,default=5)
    parser.add_argument('--speed',type=float,default=4)
    parser.add_argument('--direction',type=float,default=0)
    parser.add_argument('--threshold',type=float,default=6)
    parser.add_argument('--max-wind',type=float,default=10)
    parser.add_argument('--risk-weight',type=float,default=.2)
    parser.add_argument('--hub',type=int,nargs=2)
    parser.add_argument('--customer',type=int,nargs=2)
    parser.add_argument('--preview-3d',action='store_true')
    args=parser.parse_args()
    try:run(args)
    except (ValueError,FileNotFoundError) as exc:parser.error(str(exc))


if __name__=='__main__':main()
