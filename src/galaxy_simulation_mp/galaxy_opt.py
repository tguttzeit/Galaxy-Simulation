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
import socket
from galaxy_manager import GalaxyManager
from pathlib import Path
import time
import random
import numpy as np
from numba import jit

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
_X = 0
_Y = 1
_Z = 2

# indexes for array containing rendered data
_POS_X_RENDERED = 0
_POS_Y_RENDERED = 1
_POS_Z_RENDERED = 2
_RADIUS_RENDERED = 3
_C_RED_RENDERED = 4
_C_GREEN_RENDERED = 5
_C_BLUE_RENDERED = 6
_C_ALPHA_RENDERED = 7

# indexes for tuple containing queue output
_RESULT_INDEX = 0
_RESULT_POS = 1
_R_POS_RENDER = 2
_RESULT_VEL = 3

_PLANET_INDEX = 0
_POSITION_INDEX = 1
_VELOCITY_INDEX = 2
_DELTA_T_INDEX = 3

    
class Galaxy():
    
    def __init__(self, nr_of_bodies, start_sun_sim) -> None:
        print("initializing Galaxy...")        
        # setting simulation params
        global _START_BLACK_HOLE_SIM
        _START_BLACK_HOLE_SIM = start_sun_sim
        self.position_data = np.zeros(shape=(nr_of_bodies, 3), dtype=np.float64)
        self.velocity_data = np.zeros(shape=(nr_of_bodies, 3), dtype=np.float64)
        self.mass_data = np.zeros(shape=(nr_of_bodies, 1))
    
    def w_calculate_rendering_position(self, new_position):
        """scales the new positions that should be renderd using the current positions from body_data
            positions can only be rendered between -1 and 1, so they need to be scaled
        
        Args:
            new_position: np.array containing the real new positions (not scaled)
            
        Returns:
            np.array: calculated new positions for rendering
        """
        return new_position[_X : _Z+1] * _DISTANCE_SCALE
        
    def _calculate_acceleration(self, body_index: int, position_data):
        """calculates the current acceleration for one body regarding the forces between the bodies

        Args:
            bodyIndex: index in body_data of the body, whose acceleration should be calculated
            
        Returns:
            np.array: vector of calculated new position
        """
        totalForce = 0
        pos_target_body = position_data[body_index]
        
        for curr_body_index, body in enumerate(position_data[:]):
            if curr_body_index == body_index:
                continue
            
            pos_curr_body = body[_X:_Z+1].copy()
                
            force = _G*((self.mass_data[body_index] * self.mass_data[curr_body_index])/(np.linalg.norm(pos_curr_body - pos_target_body))**3)*(pos_curr_body - pos_target_body)
            totalForce = totalForce + force
                
        acceleration = totalForce / self.mass_data[body_index]
        return acceleration    

    def _doing_worker_task(self, job_list: list, result_list: list):
        """waits for tasks and calulates for each one the new position, new rendered position
        and velocity of one planet and puts it into the result queue.
        The result contains: body_index, new_position, rendered_position, new_velocity
    
        Args:
            p_in_queue: common job queue containing the tasks
            p_result_queue: common result_queue to put in the results of each process
        
        """   
        for task in job_list:       
            position_data = task[_POSITION_INDEX].copy()
            body_index = task[_PLANET_INDEX]
            delta_t = task[_DELTA_T_INDEX]
            
            curr_mass = self.mass_data[body_index]  
            curr_pos = position_data[body_index]
            curr_velocity = task[_VELOCITY_INDEX]
            
            accel = self._calculate_acceleration(body_index, position_data)
            new_pos = curr_pos +  curr_velocity * delta_t + 1/2 * accel * delta_t**2
            rendered_pos = self.w_calculate_rendering_position(new_pos)
            
            new_velocity = curr_velocity + accel * delta_t
            
            result = (body_index, new_pos, rendered_pos, new_velocity)
            result_list.append(result)
            #print("Task done.")
            
    def _overwrite_rendered_positions(self) -> None:
        """scales the new positions that should be renderd using the current positions from body_data
            positions can only be rendered between -1 and 1, so they need to be scaled
        """
        for body_index, body in enumerate(self.render_data[:]):
            newPosition = self.position_data[body_index]

            body[_POS_X_RENDERED : _POS_Z_RENDERED+1] = newPosition[_X : _Z+1] * _DISTANCE_SCALE
            
    def _move_bodies(self, delta_t: float) -> None:
        """
        Fills the job list and the job queue. Waits for workers to finish with job_queue.join()
        When the workers are finished, it puts everything from the result queue into the overwrite
        body_data.

        Args:
            delta_t: simulation step width
        """
        #Fill the job list with jobs containing the index of the body that the worker
        #should work on, the other planets (to be able to calc the forces) as well as delta_t.
        #index 0 is skipped as it's either the sun or the black hole in the center
        job_list = []
        for index, pos in enumerate(self.position_data):
            if index == 0:
                continue
            job_list.append((index, self.position_data, self.velocity_data[index], delta_t))
        
        result_list = []
        self._doing_worker_task(job_list, result_list)    
        
        # Empty the result queue and fill the body data array
        for result_tuple in result_list:
            body_index = result_tuple[_RESULT_INDEX]
            # Real position
            self.position_data[body_index] = result_tuple[_RESULT_POS]
            # Velocity
            self.velocity_data[body_index] = result_tuple[_RESULT_VEL]
            # Rendering position
            self.render_data[body_index][_POS_X_RENDERED : _POS_Z_RENDERED+1] = result_tuple[_R_POS_RENDER]
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
        for body_index, body in enumerate(self.render_data[:]):
            curr_body_cfg = body_data_cfg[body_index]
            
            # setting constant values
            global _G 
            _G = eval(_cfg["gravity_coefficient"])
            global _DISTANCE_SCALE 
            _DISTANCE_SCALE = eval(_cfg["distance_scale"])

            # filling body data (not the rendered array)
            # mass 
            self.mass_data[body_index] = eval(curr_body_cfg["mass"])
            
            # velocity
            self.velocity_data[body_index] = curr_body_cfg["start_v"]
            
            # position
            self.position_data[body_index] = eval(curr_body_cfg["start_p"])
            
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
        # TODO: Make this dynamic 
        total_mass = _MASS_BLACK_HOLE
        random_xy= np.random.uniform(low = _SPACE_LIMITS_XY[0], high = _SPACE_LIMITS_XY[1], size=(len(self.position_data), 2))  
        random_z = np.random.uniform(low = _SPACE_LIMITS_Z[0], high = _SPACE_LIMITS_Z[1], size=(len(self.position_data)))
        
        for index, body in enumerate(self.render_data):
            self.render_data[index][_C_ALPHA_RENDERED] = 1.0 
            if index == _POS_CENTRAL_BODY_IN_DATA:
                # data black hole
                self.mass_data[index] = _MASS_BLACK_HOLE
                self.render_data[index][_RADIUS_RENDERED] = _RADIUS_BLACK_HOLE * _RADIUS_SCALE
                #colors black hole -> grey
                self.render_data[index][_C_RED_RENDERED : _C_BLUE_RENDERED+1] = _RGB_BLACK_HOLE
                continue
            
            # initial mass
            self.mass_data[index] = random.randint(_MASS_INTERVAL[0], _MASS_INTERVAL[1]) * _MASS_MULTIPLIER
            
            # initial radius
            self.render_data[index][_RADIUS_RENDERED] = random.randint(_RADIUS_INTERVAL[0], _RADIUS_INTERVAL[1]) * _RADIUS_SCALE
            
            # start positions
            self.position_data[index][_X : _Y + 1] = random_xy[index] * _SPACE_LIMITS_XY_MULTIPLIER
            self.position_data[index][_Z] = random_z[index]         
            
            # colors bodies -> random
            self.render_data[index][_C_RED_RENDERED : _C_BLUE_RENDERED + 1] = np.random.uniform(size=3)
            
            
        total_mass = np.sum(self.mass_data)    
        for index, body in enumerate(self.velocity_data):
            # start velocity
            self._calculate_start_velocity(total_mass)
            
    
    def _calculate_start_velocity(self, total_mass) -> None:
        """Calculates the start velocities for each body except for the black hole

        Args:
            total_mass: total mass of all bodies within the system
        """
        for index, velocity in enumerate(self.velocity_data):
            if index == _POS_CENTRAL_BODY_IN_DATA:
                continue
            mass_factor = (total_mass - self.mass_data[index]) / total_mass
            
            pos_curr_body = self.position_data[index]
            center_of_mass = self._calculate_center_of_mass(total_mass, index)
            difference_pos_center_of_mass = pos_curr_body - center_of_mass
            
            velocity_abs = mass_factor * np.sqrt(_G * total_mass / np.linalg.norm(difference_pos_center_of_mass))
            cross_xyz = np.cross(np.array([0,0,1]), difference_pos_center_of_mass)
            velocity_unit_vector = cross_xyz / np.linalg.norm(cross_xyz)
            velocity = velocity_abs * velocity_unit_vector
    
    
    def _calculate_center_of_mass(self, total_mass, curr_body_index):
        """calculates the center of mass without the current body (needed to calculate the velocity)
        Args:
            total_mass: total mass of all bodies within the system
            curr_body_index: index of the body that should be excepted from the calculation

        Returns:
            np.array: center of mass without current body
        """
        sum_of_mass_pos = 0
        for index, position in enumerate(self.position_data):
            if index == curr_body_index:
                continue
            pos_curr_body = position[_X : _Z + 1].copy()
            sum_of_mass_pos = sum_of_mass_pos + self.mass_data[index] * pos_curr_body
        
        center_of_mass = 1 / (total_mass - self.mass_data[index]) * sum_of_mass_pos
        return center_of_mass   
    
    def _calculate_total_pulse_abs(self):
    #    """calculates the absolute value of the total pulse vector for the system
    #    """
    #    total_pulse = 0
    #    for body in self.body_data:
    #        total_pulse = total_pulse + body[_MASS_DATA] * body[_VELOCITY_X_DATA : _VELOCITY_Z_DATA+1]
        pass
          
    def _startup(self, shared_bodies_name: np.array, delta_t:float, step_limit):
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
               
        self.render_data = shared_bodies_name.copy()
        
        # start initialization randomly or according to config_file
        if _START_BLACK_HOLE_SIM:
            self._initialize_random_bodies()
        else:
            self._load_config_file()
            self._initialize_bodies()
        self._overwrite_rendered_positions()
        
        steps = 1
        while steps < step_limit:
            if steps % 10 == 0:
                print(steps)
            self._move_bodies(delta_t)
            steps = steps + 1
      
