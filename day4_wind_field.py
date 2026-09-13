"""Day 4: synthetic steady wind field, NOT CFD or flight guidance.
Direction means where wind travels: 0=east, 90=north (not weather FROM).
Run: python3 day4_wind_field.py --speed 4 --direction 0 --threshold 6
"""
import argparse
from dataclasses import dataclass
from math import cos, sin, radians, isfinite
from pathlib import Path
import numpy as np
from day2_city_grid import City

@dataclass(frozen=True)
class WindConfig:
    speed: float = 4.0
    direction: float = 0.0
    threshold: float = 6.0
    local_effects: bool = True

    def __post_init__(self):
        if not all(isfinite(x) for x in (self.speed, self.direction, self.threshold)):
            raise ValueError('All wind inputs must be finite.')
        if self.speed < 0 or self.threshold <= 0:
            raise ValueError('Speed must be >= 0; threshold must be > 0.')

@dataclass
class WindField:
    u: np.ndarray
    v: np.ndarray
    speed: np.ndarray
    risk: np.ndarray
    free: np.ndarray

    def at(self, city, cell):
        if not city.is_free(cell):
            raise ValueError('Wind query must be inside a free cell.')
        x, y = cell
        return dict(u=float(self.u[y,x]), v=float(self.v[y,x]),
                    speed=float(self.speed[y,x]), risk=bool(self.risk[y,x]))

def build_wind_field(city, config):
    """Rotate synthetic patches with the wind; arrays indexed [y,x].
    Lee slowdown, edge acceleration, lateral deflection are illustrative only.
    Overlapping patches use maxima rather than additive amplification.
    No temporal turbulence or actual gust prediction is represented.
    """
    angle = radians(config.direction)
    ex, ey = cos(angle), sin(angle)
    nx, ny = -ey, ex
    yy, xx = np.mgrid[0:city.rows, 0:city.columns]
    xx, yy = (xx+.5)*city.cell_m, (yy+.5)*city.cell_m
    wake = np.zeros_like(xx); edge = np.zeros_like(xx)
    side = np.zeros_like(xx)
    if config.local_effects and config.speed > 0:
        for b in city.buildings:
            cx, cy = (b.x+b.width/2)*city.cell_m, (b.y+b.height/2)*city.cell_m
            along = (xx-cx)*ex + (yy-cy)*ey
            cross = (xx-cx)*nx + (yy-cy)*ny
            half_a = (abs(ex)*b.width + abs(ey)*b.height)*city.cell_m/2
            half_c = (abs(nx)*b.width + abs(ny)*b.height)*city.cell_m/2
            lee = np.where(along >= half_a, np.exp(-np.maximum(along-half_a,0)/(3*city.cell_m)), 0)
            wake = np.maximum(wake, lee*np.exp(-(cross/(half_c+city.cell_m))**2))
            patch = np.exp(-((np.abs(cross)-half_c)/city.cell_m)**2-(along/(half_a+city.cell_m))**2)
            replace = patch > edge
            side = np.where(replace, np.sign(cross)*patch, side)
            edge = np.maximum(edge, patch)
    forward = config.speed*(1-.45*wake+.8*edge)
    lateral = config.speed*.25*side
    u, v = forward*ex+lateral*nx, forward*ey+lateral*ny
    free = np.array(city.occupancy()) == 0
    u, v = np.where(free,u,np.nan), np.where(free,v,np.nan)
    speed = np.hypot(u,v)
    risk = free & (speed > config.threshold)
    return WindField(u,v,speed,risk,free)

def save_plot(city, field, config, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    fig,ax=plt.subplots(figsize=(9,8),layout='constrained')
    extent=(0,city.columns*city.cell_m,0,city.rows*city.cell_m)
    im=ax.imshow(field.speed,origin='lower',extent=extent,cmap='YlGnBu',vmin=0,
                 vmax=max(config.threshold,float(np.nanmax(field.speed)),1))
    for b in city.buildings:
        ax.add_patch(Rectangle((b.x*city.cell_m,b.y*city.cell_m),b.width*city.cell_m,
                              b.height*city.cell_m,facecolor='#475569'))
    y,x=np.mgrid[0:city.rows:2,0:city.columns:2]
    ax.quiver((x+.5)*city.cell_m,(y+.5)*city.cell_m,field.u[::2,::2],field.v[::2,::2],
              angles='xy',scale_units='xy',scale=0.07,width=.0025,color='#111827')
    ry,rx=np.where(field.risk)
    ax.scatter((rx+.5)*city.cell_m,(ry+.5)*city.cell_m,marker='s',s=25,
               facecolors='none',edgecolors='#ef4444',linewidths=.8,label='Above demo threshold')
    for cell,label,marker,color in [(city.hub,'Hub','s','#f97316'),(city.customer,'Customer','*','#e11d48')]:
        px,py=city.cell_center_m(cell);ax.scatter(px,py,s=130,marker=marker,color=color,label=label,zorder=5)
    fig.colorbar(im,ax=ax,label='Wind speed [m/s]')
    ax.set(title=f'DAY 4 | Synthetic wind field\nBase {config.speed:g} m/s, TO {config.direction:g} deg | threshold > {config.threshold:g} m/s',xlabel='x [m] / east',ylabel='y [m] / north')
    ax.legend(loc='upper left',fontsize=8)
    fig.text(.5,.002,'Illustrative steady field; no CFD, weather data, flight certification or energy update.',ha='center',fontsize=8)
    fig.savefig(output,dpi=150);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--speed',type=float,default=4)
    p.add_argument('--direction',type=float,default=0)
    p.add_argument('--threshold',type=float,default=6)
    p.add_argument('--uniform',action='store_true')
    p.add_argument('--output',default='outputs/day4_wind_field.png')
    a=p.parse_args()
    try:
        c=City(); cfg=WindConfig(a.speed,a.direction,a.threshold,not a.uniform)
        f=build_wind_field(c,cfg)
    except ValueError as e:p.error(str(e))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    save_plot(c,f,cfg,out)
    print(f'Free cells: {f.free.sum()} | speed range: {np.nanmin(f.speed):.2f} to {np.nanmax(f.speed):.2f} m/s')
    print(f'Above demonstration threshold: {f.risk.sum()} cells; this is not a certified safety limit.')
    print('Hub:',f.at(c,c.hub));print('Customer:',f.at(c,c.customer))
    print('Saved:',out)

if __name__=='__main__':main()
