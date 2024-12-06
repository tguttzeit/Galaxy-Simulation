"""
    Module to send changing object positions through a pipe. Note that
    this is not a simulation, but a mockup.
"""
#
# Copyright (C) 2017  "Peter Roesch" <Peter.Roesch@fh-augsburg.de>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
# or open http://www.fsf.org/licensing/licenses/gpl.html
#
import json
from pathlib import Path
import time
import random
from multiprocessing.shared_memory import SharedMemory

import numpy as np

_PRINT_TOTAL_PULSE = 0

_FPS = 60
_G = 6.672*10**(-11)
_MASS_INTERVAL = (1, 10)
_MASS_MULTIPLIER = 10**27
_RADIUS_INTERVAL = (2400, 40250)
_MAX_DISTANCE = 10**12
_DISTANCE_SCALE = 0.98/_MAX_DISTANCE
_RADIUS_SCALE = 1000 * 0.07/(5*10**7)
_SPACE_LIMITS_XY_MULTIPLIER = 10**11
_SPACE_LIMITS_XY = (-10, 10)
_SPACE_LIMITS_Z = (-1000, 1000)

_MASS_BLACK_HOLE = 50*10**(30)
_RADIUS_BLACK_HOLE = 10000
_RGB_BLACK_HOLE = np.array([0.25, 0.25, 0.25])

# indexes for array containing body data
_POS_CENTRAL_BODY_IN_DATA = 0
_POS_X_DATA = 0
_POS_Y_DATA = 1
_POS_Z_DATA = 2
_MASS_DATA = 3
_VELOCITY_X_DATA = 4
_VELOCITY_Y_DATA = 5
_VELOCITY_Z_DATA = 6

# indexes for array containing rendered data
_POS_X_RENDERED = 0
_POS_Y_RENDERED = 1
_POS_Z_RENDERED = 2
_RADIUS_RENDERED = 3
_C_RED_RENDERED = 4
_C_GREEN_RENDERED = 5
_C_BLUE_RENDERED = 6
_C_ALPHA_RENDERED = 7

class Galaxy():
    
    def __init__(self, nr_of_bodies, start_sun_sim) -> None:
        global _START_SUN_SIM
        _START_SUN_SIM = start_sun_sim
        self.body_data = np.zeros(shape=(nr_of_bodies, 7))
        pass
    
    def _calculate_acceleration(self, body_index: int):
        """calculates the current acceleration for one body regarding the forces between the bodies

        Args:
            bodyIndex: index in body_data of the body, whose acceleration should be calculated
        
        Returns:
            np.array: vector of calculated new position
        """
        totalForce = 0
        central_body = self.body_data[body_index]
        
        for currbody_index, body in enumerate(self.body_data[:]):
            if currbody_index == body_index:
                continue
            
            pos_central_body = central_body[_POS_X_DATA:_POS_Z_DATA+1].copy()
            pos_curr_body = body[_POS_X_DATA:_POS_Z_DATA+1].copy()
            
            force = _G*((central_body[_MASS_DATA] * body[_MASS_DATA])/(np.linalg.norm(pos_curr_body - pos_central_body))**3)*(pos_curr_body - pos_central_body)
            totalForce = totalForce + force
            
        acceleration = totalForce / central_body[_MASS_DATA]
        return acceleration

    def _calculate_new_position(self, body_index: int, delta_t: float):
        """calculates new positon for one body and sets new velocity according to calculated acceleration

        Args:
            bodyIndex: index in body_data of the body, whose position should be calculated
            delta_t: simulation step width

        Returns:
            np.array: vector of calculated new position
        """
        
        # skip the sun
        if body_index == _POS_CENTRAL_BODY_IN_DATA:
            return self.body_data[body_index][_POS_X_DATA:_POS_Z_DATA+1]
        
        curr_body = self.body_data[body_index]  
        curr_pos = curr_body[_POS_X_DATA:_POS_Z_DATA+1].copy()
        curr_velocity = curr_body[_VELOCITY_X_DATA:_VELOCITY_Z_DATA+1].copy()
        
        accel = self._calculate_acceleration(body_index)
        newPos = curr_pos +  curr_velocity * delta_t + 1/2 * accel * delta_t**2
        # setting new velocity
        curr_body[_VELOCITY_X_DATA:_VELOCITY_Z_DATA+1] = curr_velocity + accel * delta_t
        return newPos

    def _overwrite_rendered_positions(self) -> None:
        """scales the new positions that should be renderd using the current positions from body_data
            positions can only be rendered between -1 and 1, so they need to be scaled
        """
        for body_index, body in enumerate(self.render_data[:]):
            newPosition = self.body_data[body_index]

            body[_POS_X_RENDERED : _POS_Z_RENDERED+1] = newPosition[_POS_X_DATA : _POS_Z_DATA+1] * _DISTANCE_SCALE
            #print(body[_POS_X_RENDERED : _POS_Z_RENDERED+1])
            
    def _move_bodies(self, delta_t: float) -> None:
        """calculates next position for each body and after that saves the new positions in both given arrays

        Args:
            delta_t: simulation step width
        """
        new_positions = np.zeros((len(self.body_data), 3), dtype=np.float32)
        for body_index, body in enumerate(self.body_data[:]):
            newPos = self._calculate_new_position(body_index, delta_t)
            new_positions[body_index] = newPos

        # setting new positions
        for pos_index, pos in enumerate(new_positions):
            self.body_data[pos_index][_POS_X_DATA : _POS_Z_DATA+1] = pos.copy()
        
        time.sleep(1 / _FPS)

    def _load_config_file(self):
        cfg_path = f"{Path(__file__, '..').resolve()}/galaxy_cfg.json"
        with open (cfg_path) as cfg_file:
            global _cfg
            _cfg = json.load(cfg_file)

    def _initialize_bodies(self) -> None:
        """
        Get the position, mass, velocity out of the config file and put it into the 7-tupel for one body
        Save the start position, radius and color values for each body into the rendered array
        Set constant values like gravity coefficient and the distance scale
        """
        body_data_cfg = _cfg["planets"]
        
        #format body_data: (pos_x, pos_y, pos_z, mass, vel_x, vel_y, vel_z)
        for body_index, body in enumerate(self.body_data[:]):
            curr_body_cfg = body_data_cfg[body_index]
            
            # setting constant values
            global _G 
            _G = eval(_cfg["gravity_coefficient"])
            global _DISTANCE_SCALE 
            _DISTANCE_SCALE = eval(_cfg["distance_scale"])

            # filling body data (not the rendered array)
            # mass 
            body[_MASS_DATA] = eval(curr_body_cfg["mass"])
            
            # velocity
            velocity = curr_body_cfg["start_v"]
            body[_VELOCITY_X_DATA:_VELOCITY_Z_DATA+1] = velocity
            
            # position
            curr_pos = eval(curr_body_cfg["start_p"])
            body[_POS_X_DATA:_POS_Z_DATA+1] = curr_pos
            
            # setting data for rendering into the rendered array
            # Radius
            # format rendered body array: (pos_x, pos_y, pos_z, radius, c_red, c_green, c_blue, c_alpha)
            self.render_data[body_index][_RADIUS_RENDERED] = curr_body_cfg["radius"] * eval(_cfg["radius_scale"])
            
            # Color (RGB values)
            curr_rgb = curr_body_cfg["c_rgb"]
            self.render_data[body_index][_C_RED_RENDERED] = curr_rgb[0]
            self.render_data[body_index][_C_GREEN_RENDERED] = curr_rgb[1]
            self.render_data[body_index][_C_BLUE_RENDERED] = curr_rgb[2]
            self.render_data[body_index][_C_ALPHA_RENDERED] = 1.0        
            
    def _initialize_random_bodies(self) -> None:
        """
        Set data for all bodies randomly into the body_data array. Masses are chosen randomly within an interval.
        The start positions are set on the x and y axes according to the position in the array. They all have the same distance between them.
        Calculates the start velocities
        Save the radius and color values for each body into the rendered array
        """
        total_mass = _MASS_BLACK_HOLE
        random_xy= np.random.uniform(low = _SPACE_LIMITS_XY[0], high = _SPACE_LIMITS_XY[1], size=(len(self.body_data), 2))  
        random_z = np.random.uniform(low = _SPACE_LIMITS_Z[0], high = _SPACE_LIMITS_Z[1], size=(len(self.body_data)))
        
        for index, body in enumerate(self.body_data):
            self.render_data[index][_C_ALPHA_RENDERED] = 1.0 
            if index == _POS_CENTRAL_BODY_IN_DATA:
                # data black hole
                body[_MASS_DATA] = _MASS_BLACK_HOLE
                self.render_data[index][_RADIUS_RENDERED] = _RADIUS_BLACK_HOLE * _RADIUS_SCALE
                #colors black hole -> grey
                self.render_data[index][_C_RED_RENDERED : _C_BLUE_RENDERED+1] = _RGB_BLACK_HOLE
                continue
            
            # initial mass
            body[_MASS_DATA] = random.randint(_MASS_INTERVAL[0], _MASS_INTERVAL[1]) * _MASS_MULTIPLIER
            total_mass = total_mass + body[_MASS_DATA]
            
            # initial radius
            self.render_data[index][_RADIUS_RENDERED] = random.randint(_RADIUS_INTERVAL[0], _RADIUS_INTERVAL[1]) * _RADIUS_SCALE
            
       
            # start positions
            body[_POS_X_DATA : _POS_Y_DATA+1] = random_xy[index] * _SPACE_LIMITS_XY_MULTIPLIER
            body[_POS_Z_DATA] = random_z[index]         
            
            # colors bodies -> random
            self.render_data[index][_C_RED_RENDERED : _C_BLUE_RENDERED+1] = np.random.uniform(size=3)
            
            # start velocity
            self._calculate_start_velocity(total_mass)
    
    def _calculate_start_velocity(self, total_mass) -> None:
        """Calculates the start velocities for each body except for the black hole

        Args:
            total_mass: total mass of all bodies within the system
        """
        for index, body in enumerate(self.body_data):
            if index == _POS_CENTRAL_BODY_IN_DATA:
                continue
            mass_factor = (total_mass - body[_MASS_DATA]) / total_mass
            
            pos_curr_body = body[_POS_X_DATA:_POS_Z_DATA+1].copy()
            center_of_mass = self._calculate_center_of_mass(total_mass, index)
            difference_pos_center_of_mass = pos_curr_body - center_of_mass
            
            velocity_abs = mass_factor * np.sqrt(_G * total_mass / np.linalg.norm(difference_pos_center_of_mass))
            velocity_unit_vector = self._calculate_velocity_unit_vector(difference_pos_center_of_mass)
            body[_VELOCITY_X_DATA : _VELOCITY_Z_DATA+1] = velocity_abs * velocity_unit_vector
    
    def _calculate_center_of_mass(self, total_mass, curr_body_index):
        """calculates the center of mass without the current body (needed to calculate the velocity)
        Args:
            total_mass: total mass of all bodies within the system
            curr_body_index: index of the body that should be excepted from the calculation

        Returns:
            np.array: center of mass without current body
        """
        sum_of_mass_pos = 0
        for index, body in enumerate(self.body_data):
            if index == curr_body_index:
                continue
            pos_curr_body = body[_POS_X_DATA:_POS_Z_DATA+1].copy()
            sum_of_mass_pos = sum_of_mass_pos + body[_MASS_DATA] * pos_curr_body
        
        center_of_mass = 1 / (total_mass - body[_MASS_DATA]) * sum_of_mass_pos
        return center_of_mass
    
    def _calculate_velocity_unit_vector(self, difference_pos_center_of_mass):
        """calculates the unit vector of the velocity of one body
           needed to get z coord

        Args:
            difference_pos_center_of_mass: result vector of current_position - center_of_mass

        Returns:
            np.array: calculated unit vector of the velocity
        """
        z = np.array([0,0,1])
        cross_xyz = np.cross(z, difference_pos_center_of_mass)
        velocity_unit_vector = cross_xyz / np.linalg.norm(cross_xyz)
        return velocity_unit_vector
    
    def _calculate_total_pulse_abs(self):
        """calculates the absolute value of the total pulse vector for the system
        """
        total_pulse = 0
        for body in self.body_data:
            total_pulse = total_pulse + body[_MASS_DATA] * body[_VELOCITY_X_DATA : _VELOCITY_Z_DATA+1]
        print(np.linalg.norm(total_pulse))
            
    def startup(self, shared_bodies_name: str, shared_flags_name: str, delta_t:float):
        """
        Load values from config file
        Initialise and continuously update a position list.
        Initialize a list with all necessary data about the bodies
        Results are sent through a pipe after each update step

        Args:
            shared_bodies_name: Name of bodies shared memory
            shared_flags_name: Name of flags shared memory
            delta_t: Simulation step width.
        """
               
        flags_shm = SharedMemory(shared_flags_name)
        shared_flags = flags_shm.buf
        bodies_shm = SharedMemory(shared_bodies_name)
        float32_nr = len(bodies_shm.buf) // 4
        self.render_data = np.ndarray(
            shape=(float32_nr,), dtype=np.float32, buffer=bodies_shm.buf
        )
        self.render_data = self.render_data.reshape(-1, 8)
        
        # start initialization randomly or according to config_file
        if _START_SUN_SIM:
            self._initialize_random_bodies()
        else:
            self._load_config_file()
            self._initialize_bodies()
        
        self._overwrite_rendered_positions()
        time.sleep(2)
        
        while not shared_flags[1]:
            self._move_bodies(delta_t)
            self._overwrite_rendered_positions()
            if _PRINT_TOTAL_PULSE:
                self._calculate_total_pulse_abs()
            shared_flags[0] = 1
        for s in (flags_shm, bodies_shm):
            s.close()
