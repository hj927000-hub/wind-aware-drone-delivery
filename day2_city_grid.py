"""Day 2: synthetic occupancy grid, not real SF geography or flight guidance.

Run: python3 day2_city_grid.py
Install plotting dependency: python3 -m pip install matplotlib
Coordinates are (column, row); occupancy is grid[row][column].
Buildings block the entire modeled flight layer. No heights, wind or routing.
"""
from dataclasses import dataclass
from math import hypot, isfinite
from pathlib import Path


@dataclass(frozen=True)
class Building:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class City:
    columns: int = 40
    rows: int = 40
    cell_m: float = 50.0
    hub: tuple[int, int] = (4, 4)
    customer: tuple[int, int] = (34, 34)
    buildings: tuple[Building, ...] = (
        Building(8, 5, 5, 9), Building(17, 2, 6, 8),
        Building(27, 6, 7, 6), Building(3, 20, 6, 7),
        Building(13, 17, 8, 8), Building(25, 19, 5, 10),
        Building(10, 31, 8, 5), Building(22, 33, 6, 5),
    )

    def __post_init__(self):
        for n in (self.columns, self.rows):
            if type(n) is not int or n <= 0:
                raise ValueError('Grid dimensions must be positive integers.')
        if not isfinite(self.cell_m) or self.cell_m <= 0:
            raise ValueError('Cell size must be positive and finite.')
        for b in self.buildings:
            if any(type(v) is not int for v in (b.x, b.y, b.width, b.height)):
                raise ValueError('Building geometry must use integer cells.')
            if b.width <= 0 or b.height <= 0 or b.x < 0 or b.y < 0:
                raise ValueError('Invalid building rectangle.')
            if b.x + b.width > self.columns or b.y + b.height > self.rows:
                raise ValueError('Building extends outside map.')
        for cell in (self.hub, self.customer):
            if not self.is_free(cell):
                raise ValueError('Hub and customer must be inside free cells.')

    def inside(self, cell):
        x, y = cell
        return (type(x) is int and type(y) is int
                and 0 <= x < self.columns and 0 <= y < self.rows)

    def is_free(self, cell):
        if not self.inside(cell):
            return False
        x, y = cell
        return not any(b.x <= x < b.x + b.width and
                       b.y <= y < b.y + b.height for b in self.buildings)

    def occupancy(self):
        return [[int(not self.is_free((x, y))) for x in range(self.columns)]
                for y in range(self.rows)]

    def cell_center_m(self, cell):
        if not self.inside(cell):
            raise ValueError('Cell outside map.')
        x, y = cell
        return ((x + 0.5) * self.cell_m, (y + 0.5) * self.cell_m)

    def straight_distance_m(self):
        a, b = self.cell_center_m(self.hub), self.cell_center_m(self.customer)
        return hypot(b[0] - a[0], b[1] - a[1])


def save_map(city, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
    extent = (0, city.columns * city.cell_m, 0, city.rows * city.cell_m)
    ax.imshow(city.occupancy(), origin='lower', extent=extent,
              cmap=ListedColormap(['#f1f5f9', '#475569']), vmin=0, vmax=1,
              interpolation='nearest')
    for cell, label, color, marker in (
        (city.hub, 'Hub', '#0284c7', 's'),
        (city.customer, 'Customer', '#e17020', '*'),
    ):
        x, y = city.cell_center_m(cell)
        ax.scatter(x, y, color=color, marker=marker, s=180, zorder=4, label=label)
        ax.annotate(f'{label} {cell}', (x, y), xytext=(10, -20),
                    textcoords='offset points', fontsize=10)
    # Intentionally no line: a straight line would intersect obstacles.
    ax.set_xticks(range(0, int(extent[1]) + 1, 250))
    ax.set_yticks(range(0, int(extent[3]) + 1, 250))
    ax.set_xticks([i * city.cell_m for i in range(city.columns + 1)], minor=True)
    ax.set_yticks([i * city.cell_m for i in range(city.rows + 1)], minor=True)
    ax.grid(which='minor', color='#cbd5e1', linewidth=0.35)
    ax.set_xlabel('Local x [m]')
    ax.set_ylabel('Local y [m]')
    ax.set_title('DAY 2 | Synthetic city occupancy grid\n'
                 f'{city.columns} x {city.rows} cells | {city.cell_m:g} m per cell',
                 loc='left', fontsize=14, pad=15)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(color='#475569', label='Blocked building footprint'))
    ax.legend(handles=handles, loc='upper left', fontsize=9)
    fig.text(0.5, 0.005, 'Not real SF data. No route, altitude or airspace clearance.',
             ha='center', fontsize=9, color='#64748b')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    city = City()
    path = Path(__file__).resolve().parent / 'outputs/day2_city_map.png'
    save_map(city, path)
    blocked = sum(map(sum, city.occupancy()))
    print(f'Map: {city.columns * city.cell_m:g} x {city.rows * city.cell_m:g} m')
    print(f'Cells: {city.columns * city.rows}; blocked: {blocked}')
    print(f'Hub center [m]: {city.cell_center_m(city.hub)}')
    print(f'Customer center [m]: {city.cell_center_m(city.customer)}')
    print(f'Straight distance: {city.straight_distance_m():.2f} m (NOT a route)')
    print('Day 1 preserved. No obstacle-aware energy estimate until Day 3.')
    print(f'Map saved: {path}')


if __name__ == '__main__':
    main()
