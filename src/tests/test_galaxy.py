""" Tests for module galaxy"""

import json
import numpy as np
import unittest
import galaxy_simulation.galaxy as galaxy
from unittest import mock


_SUN = 0
_MERCURY = 1
_VENUS = 2

class TestGalaxy(unittest.TestCase):    
    def setUp(self):
        self._solarsystem = galaxy.Galaxy()
        self.correct_cfg = json.loads('{"distance_scale": "0.98/(4539.24*10**(9))","radius_scale": "0.07/69911000","gravity_coefficient": "6.672*10**(-11)","delta_t": 30000,"planets": [{"name": "sun","c_rgb": [1.0, 0.835, 0],"mass": "1.98*10**(30)","radius": 6963400,"start_p": "[0,0,0]","start_v": [0, 0, 0]},{"name": "mercury","c_rgb": [0.557,0.529,0.467],"mass": "3.301*10**(23)","radius": 2439770,"start_p": "[46.001*10**(9),0,0]","start_v": [0,53078.4,25707.04241]},{"name": "venus","c_rgb": [0.835,0.733,0.494],"mass": "4.875*10**(24)","radius": 6051800,"start_p": "[107.473*10**(9),0,0]","start_v": [0,35260,0]}]}')
        galaxy._G = eval(self.correct_cfg["gravity_coefficient"])
        galaxy._DISTANCE_SCALE = eval(self.correct_cfg["distance_scale"])
        
        sun_data = self.correct_cfg["planets"][_SUN]
        mercury_data = self.correct_cfg["planets"][_MERCURY]
        venus_data = self.correct_cfg["planets"][_VENUS]
        self.correct_d_scale = eval(self.correct_cfg["distance_scale"])
        self.correct_r_scale = eval(self.correct_cfg["radius_scale"])
        
        # Set up the rendering array before overwritten positions
        self.correct_render_data = np.array([
            [0,0,0,
             sun_data["radius"]*self.correct_r_scale,
             sun_data["c_rgb"][0],sun_data["c_rgb"][1],sun_data["c_rgb"][2],1.0],

            [0,0,0,
             mercury_data["radius"]*self.correct_r_scale,
             mercury_data["c_rgb"][0],mercury_data["c_rgb"][1],mercury_data["c_rgb"][2],1.0],
            
            [0,0,0,
             venus_data["radius"]*self.correct_r_scale,
             venus_data["c_rgb"][0],venus_data["c_rgb"][1],venus_data["c_rgb"][2],1.0]])     
        
        # Set up the rendering array after overwritten positions
        self.correct_render_data_with_pos = np.array([
            [eval(sun_data["start_p"])[0]*self.correct_d_scale,
             eval(sun_data["start_p"])[1]*self.correct_d_scale,
             eval(sun_data["start_p"])[2]*self.correct_d_scale,
             sun_data["radius"]*self.correct_r_scale,
             sun_data["c_rgb"][0],sun_data["c_rgb"][1],sun_data["c_rgb"][2],1.0],

            [eval(mercury_data["start_p"])[0]*self.correct_d_scale,
             eval(mercury_data["start_p"])[1]*self.correct_d_scale,
             eval(mercury_data["start_p"])[2]*self.correct_d_scale,
             mercury_data["radius"]*self.correct_r_scale,
             mercury_data["c_rgb"][0],mercury_data["c_rgb"][1],mercury_data["c_rgb"][2],1.0],
            
            [eval(venus_data["start_p"])[0]*self.correct_d_scale,
             eval(venus_data["start_p"])[1]*self.correct_d_scale,
             eval(venus_data["start_p"])[2]*self.correct_d_scale,
             venus_data["radius"]*self.correct_r_scale,
             venus_data["c_rgb"][0],venus_data["c_rgb"][1],venus_data["c_rgb"][2],1.0]])     
        
        # Setup the planet array with realistic values for calculation
        self.correct_planet_data = np.array([
            [eval(sun_data["start_p"])[0],
             eval(sun_data["start_p"])[1],
             eval(sun_data["start_p"])[2],
             eval(sun_data["mass"]),
             sun_data["start_v"][0],sun_data["start_v"][1],sun_data["start_v"][2]],

            [eval(mercury_data["start_p"])[0],
             eval(mercury_data["start_p"])[1],
             eval(mercury_data["start_p"])[2],
             eval(mercury_data["mass"]),
             mercury_data["start_v"][0],mercury_data["start_v"][1],mercury_data["start_v"][2]],
            
            [eval(venus_data["start_p"])[0],
             eval(venus_data["start_p"])[1],
             eval(venus_data["start_p"])[2],
             eval(venus_data["mass"]),
             venus_data["start_v"][0],venus_data["start_v"][1],venus_data["start_v"][2]]]) 
        
    def test_load_cfg_file(self) -> None:
        self.setUp()
        self._solarsystem._load_config_file()
        self.config_data = galaxy._cfg
        self.assertEqual(self.correct_cfg["distance_scale"], self.config_data["distance_scale"])
        self.assertEqual(self.correct_cfg["radius_scale"], self.config_data["radius_scale"])
        self.assertEqual(self.correct_cfg["gravity_coefficient"], self.config_data["gravity_coefficient"])
        self.assertEqual(self.correct_cfg["delta_t"], self.config_data["delta_t"])
        self.assertEqual(self.correct_cfg["planets"][0], self.config_data["planets"][0])
        print()
        
    def test_initialize_planets(self) -> None:
        self.setUp()
        test_render_data = np.zeros((3, 8))        
        test_planet_data = np.zeros((3, 7))
        galaxy._cfg = self.correct_cfg
        self._solarsystem._initialize_planets(test_render_data, test_planet_data)
        np.testing.assert_array_equal(test_planet_data, self.correct_planet_data)
        np.testing.assert_array_equal(test_render_data, self.correct_render_data)
        
    def test_overwrite_rendered_positions(self) -> None:
        self.setUp()
        test_render_data = self.correct_render_data.copy()
        galaxy._overwrite_rendered_positions(test_render_data, self.correct_planet_data)
        np.testing.assert_array_equal(self.correct_render_data_with_pos, test_render_data)
        
    def test_calculate_acceleration(self) -> None:
        self.setUp()
        correct_mercury_acceleration = [-0.062429, 0, 0]
        epsilon = 5
        test_mercury_acceleration = self._solarsystem._calculate_acceleration(_MERCURY, self.correct_planet_data)
        np.testing.assert_array_almost_equal(test_mercury_acceleration, correct_mercury_acceleration, epsilon) 
        
    def test_calculate_new_position(self) -> None:
        self.setUp()
        correct_mercury_position = eval("[4.59729*10**(3), 1.59235*10**(2), 7.71211*10]")
        epsilon = 2
        test_mercury_position = self._solarsystem._calculate_new_position(_MERCURY, self.correct_planet_data, self.correct_cfg["delta_t"])*10**(-7)
        np.testing.assert_array_almost_equal(test_mercury_position, correct_mercury_position, epsilon) 
                
    
if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGalaxy)
    unittest.TextTestRunner(verbosity=1).run(suite)
    

