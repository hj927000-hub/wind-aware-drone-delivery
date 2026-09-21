"""Geometry and existing-routing integration checks; no network calls."""
import json
from math import hypot
from pathlib import Path
import unittest

import numpy as np
from pyproj import Transformer
from shapely.geometry import box, LineString, Polygon
from shapely.ops import unary_union

from day3_astar_route import astar
from day4_wind_field import WindConfig, build_wind_field
from day5_wind_energy_model import FlightConfig, evaluate
from day6_wind_aware_routing import weighted_astar
from day8_manhattan import (load_buildings, rasterize, RealCity, largest_component,
                            demo_endpoints, parse_height_m)


def blocked_set(grid):
    return {(int(x),int(y)) for y,x in np.argwhere(grid)}


class Day8Tests(unittest.TestCase):
    def test_meter_projection_and_height_provenance(self):
        buildings,meta=load_buildings(Path(__file__).parent/'data')
        fwd=Transformer.from_crs(4326,meta['metric_crs'],always_xy=True)
        x,y=fwd.transform(*meta['center_lon_lat']);ox,oy=meta['origin_utm_m']
        self.assertAlmostEqual(x-ox,500);self.assertAlmostEqual(y-oy,500)
        self.assertGreater(len(buildings),100)
        self.assertAlmostEqual(parse_height_m('100 ft'),30.48)
        self.assertIsNone(parse_height_m(None));self.assertIsNone(parse_height_m('20;30'))
        self.assertTrue(all(p['height_status']=='unknown' for _,p in buildings if p['height_m'] is None))

    def test_raster_preserves_holes_and_detects_cell_edge_overlap(self):
        p=Polygon([(10,10),(50,10),(50,50),(10,50)],holes=[[(20,20),(20,40),(40,40),(40,20)]])
        grid=rasterize(p,60,5,0)
        self.assertFalse(grid[5,5])  # entire cell is in courtyard
        self.assertTrue(grid[2,2])   # building
        sliver=box(9.1,1,9.9,9)
        self.assertTrue(rasterize(sliver,20,10,0)[0,0]) # center-only raster would miss this
        self.assertGreaterEqual(rasterize(p,60,5,3).sum(),grid.sum())
        with self.assertRaises(ValueError):rasterize(p,60,7,0)

    def test_no_route_through_complete_barrier(self):
        wall=box(45,0,55,100)
        grid=rasterize(wall,100,10,0);city=RealCity(grid,10)
        self.assertIsNone(astar(city,(1,4),(8,4),blocked_set(grid)))

    def test_real_route_clearance_and_energy_optimality(self):
        buildings,meta=load_buildings(Path(__file__).parent/'data')
        geometry=unary_union([g for g,_ in buildings])
        raw=rasterize(geometry,meta['side_m'],5,0)
        blocked=blocked_set(rasterize(geometry,meta['side_m'],5,5))
        city=RealCity(raw,5)
        city.hub,city.customer=demo_endpoints(city,largest_component(city,blocked))
        field=build_wind_field(city,WindConfig(4,0,6,False));cfg=FlightConfig()
        a=weighted_astar(city,field,city.hub,city.customer,blocked,1.5,cfg)
        d=weighted_astar(city,field,city.hub,city.customer,blocked,1.5,cfg,False)
        self.assertIsNotNone(a)
        a_cost=sum(r['cost_wh'] for r in evaluate(city,field,a,1.5,cfg))
        d_cost=sum(r['cost_wh'] for r in evaluate(city,field,d,1.5,cfg))
        self.assertAlmostEqual(a_cost,d_cost,places=8)
        for first,second in zip(a,a[1:]):
            segment=LineString([city.cell_center_m(first),city.cell_center_m(second)])
            self.assertFalse(segment.intersects(geometry.buffer(5)))
        self.assertTrue(all(c not in blocked for c in a))


if __name__=='__main__':unittest.main()
