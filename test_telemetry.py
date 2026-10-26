"""Regression tests; run python -m unittest -v test_telemetry."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import analyze as a
import acc_reader as reader

def samples(number=1, duration=100000, count=101):
    return pd.DataFrame(dict(timestamp=np.linspace(0,100,count), lap_number=number,
        lap_time=np.linspace(0,duration,count), speed_kmh=150, throttle=.5, brake=0,
        steering=0, track_position=np.linspace(.001,.999,count)))

class Regressions(unittest.TestCase):
    def test_player_id_is_not_array_index(self):
        g=NS(car_id=[12,47], player_car_id=47,
             car_coordinates=[NS(x=1,y=2,z=3),NS(x=4,y=5,z=6)])
        self.assertEqual(reader.player_coordinates(g),(4,5,6))
        g.player_car_id=99
        self.assertEqual(reader.player_coordinates(g),(None,None,None))
    def test_numeric_live_status(self):
        self.assertTrue(reader.is_live(NS(status=2)))
        self.assertFalse(reader.is_live(NS(status=1)))
    def test_unknown_validity(self):
        self.assertIsNone(a.to_bool("unknown"))
        self.assertIs(a.to_bool("0"),False)
    def test_partial_final_lap_not_complete(self):
        self.assertFalse(a.find_laps(samples())[0].complete)
    def test_line_crossing_completes_previous_lap(self):
        d=pd.concat([samples(),samples(number=2).iloc[:2]],ignore_index=True)
        self.assertTrue(a.find_laps(d)[0].complete)
    def test_progress_not_stretched(self):
        info=a.LapInfo(1,samples(),100000,True,True)
        lap=a.build_lap(info,True)
        self.assertAlmostEqual(lap.progress.iloc[0],.001)
        self.assertAlmostEqual(lap.progress.iloc[-1],.999)
    def test_delta_and_sections_match_official_times(self):
        r=a.build_lap(a.LapInfo(1,samples(),100000,True,True),True)
        l=a.build_lap(a.LapInfo(2,samples(duration=101000),101000,True,True),True)
        delta=a.compute_delta(l,r,np.linspace(0,1,1000))
        self.assertAlmostEqual(delta[0],0)
        self.assertAlmostEqual(delta[-1],1)
        self.assertAlmostEqual(a.analyze_sections(l,r,10).time_difference.sum(),1)
    def test_zero_progress_rejected(self):
        d=samples()
        d.track_position=.1
        with self.assertRaises(ValueError):
            a.build_lap(a.LapInfo(1,d,100000,True,True),True)
    def test_known_valid_preferred_to_unknown(self):
        d=samples()
        laps=[a.LapInfo(1,d,90000,None,True),a.LapInfo(2,d,100000,True,True)]
        _,ref=a.choose_laps(laps,1,None)
        self.assertEqual(ref.number,2)
    def test_profile_has_all_16_turns(self):
        profile=a.load_profile("",Path(__file__).parent/"profiles/barcelona.json")
        ref=a.build_lap(a.LapInfo(1,samples(),100000,True,True),True)
        ref["dist_m"]=ref.progress*ref.attrs["raw_length_m"]
        corners=a.profile_corners(ref,a.detect_braking(ref),profile)
        self.assertEqual(corners.corner.tolist(),list(range(1,17)))
        self.assertLess(corners.iloc[14].end_dist,corners.iloc[15].start_dist)
        fig=a.plot_overview(ref,ref,1,1,np.linspace(0,ref.dist_m.iloc[-1],100),np.zeros(100),corners)
        self.assertEqual([t.get_text() for t in fig.axes[0].texts],[f"T{i}" for i in range(1,17)])
        a.plt.close(fig)
    def test_invalid_profile_rejected(self):
        with self.assertRaises(ValueError):
            a.validate_profile(dict(turns=[dict(start=.1,apex=.05,end=.2)]))
    def test_missing_required_data_error(self):
        from io import StringIO
        with self.assertRaises(ValueError):
            a.load_telemetry(StringIO("lap_number\n1\n"))
    def test_legacy_without_coordinates(self):
        d=samples().drop(columns="track_position")
        lap=a.build_lap(a.LapInfo(1,d,100000,True,True),False)
        self.assertAlmostEqual(lap.progress.iloc[-1],1)
    def test_no_corners_safe(self):
        ref=a.build_lap(a.LapInfo(1,samples(),100000,True,True),True)
        ref["dist_m"]=ref.progress*ref.attrs["raw_length_m"]
        self.assertTrue(a.detect_corners(ref,a.detect_braking(ref)).empty)

if __name__=="__main__":
    unittest.main()
