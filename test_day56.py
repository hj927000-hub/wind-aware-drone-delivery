import unittest
from math import isclose
from day5_wind_energy_model import *
from day6_wind_aware_routing import *
from day1_drone_mission import simulate_mission,Mission
class Tests(unittest.TestCase):
    def test_hand_cases(self):
        for u,v,expected in [(0,0,12),(4,0,16),(-4,0,8),(0,4,(144-16)**.5)]:
            r=segment(1000,1,0,u,v);self.assertAlmostEqual(r['ground_mps'],expected);self.assertAlmostEqual(r['energy_wh'],1430*1000/expected/3600)
    def test_no_progress_and_hard_limit(self):
        self.assertIsNone(segment(1000,1,0,11,0))
        cfg=FlightConfig(max_wind=20)
        self.assertIsNone(segment(1000,1,0,-12,0,cfg=cfg))
        self.assertIsNone(segment(1000,1,0,0,12,cfg=cfg))
    def test_zero_wind_day1(self):
        c,f,cfg,paths,res,_=run(speed=0)
        old=simulate_mission(Drone(),Mission(one_way_distance_km=res['distance']['round_trip_m']/2000),Economics())
        self.assertAlmostEqual(old['total_energy_wh'],res['distance']['energy_wh'])
        self.assertAlmostEqual(res['distance']['energy_wh'],res['wind']['energy_wh'])
    def test_astar_matches_dijkstra(self):
        for speed,rw in [(4,0),(6,5)]:
            c,f,cfg,paths,res,_=run(speed=speed,risk_weight=rw)
            blocked=blocked_cells(c,1)
            for start,goal,payload,p in [(c.hub,c.customer,1.5,paths['wind'][0]),(c.customer,c.hub,0,paths['wind'][1])]:
                oracle=weighted_astar(c,f,start,goal,blocked,payload,cfg,False)
                cost=lambda path:sum(r['cost_wh'] for r in evaluate(c,f,path,payload,cfg))
                self.assertAlmostEqual(cost(p),cost(oracle))
                for a,b in zip(p,p[1:]):self.assertIn(b,dict(neighbors(c,a,blocked)))
            if 'cost_wh' in res['distance']:self.assertLessEqual(res['wind']['cost_wh'],res['distance']['cost_wh']+1e-8)
    def test_no_route(self):
        self.assertFalse(run(buffer=2)[4]['wind']['feasible'])
        self.assertFalse(run(speed=13,uniform=True)[4]['wind']['feasible'])
    def test_risk_monotonic(self):
        low=run(speed=6,risk_weight=0)[4]['wind'];high=run(speed=6,risk_weight=5)[4]['wind']
        self.assertLessEqual(high['risk_s'],low['risk_s']+1e-8)
    def test_direction_asymmetry(self):
        self.assertLess(segment(1000,1,0,4,0)['energy_wh'],segment(1000,-1,0,4,0)['energy_wh'])
    def test_validation(self):
        for kw in [dict(time_weight=-1),dict(airspeed=0),dict(risk_weight=float('nan'))]:
            with self.assertRaises(ValueError):FlightConfig(**kw)
if __name__=='__main__':unittest.main()
