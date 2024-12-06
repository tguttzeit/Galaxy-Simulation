from multiprocessing import Process, cpu_count

from galaxy_manager import GalaxyManager
from numba import jit
import galaxy as g

import numpy as np
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

_PLANET_INDEX = 0
_POSITION_INDEX = 1
_VELOCITY_INDEX = 2
_DELTA_T_INDEX = 3

class GalaxyWorker():
    
    def __init__(self, manager_ip, manager_port) -> None:
        warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
        warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
        warnings.simplefilter('ignore', category=NumbaWarning)
        self.mass_initialized = False
        GalaxyManager.register('get_job_queue')
        GalaxyManager.register('get_result_queue')
        GalaxyManager.register('get_mass_data')
        self.m = GalaxyManager(address=(manager_ip, manager_port), authkey = b'secret')
        self.m.connect()
        print("sucessfully connected", self.m.address)
        self._start_workers()
        
    def _calculate_rendering_position(self, new_position):
            """scales the new positions that should be renderd using the current positions from body_data
                positions can only be rendered between -1 and 1, so they need to be scaled
            
            Args:
                new_position: np.array containing the real new positions (not scaled)
                
            Returns:
                np.array: calculated new positions for rendering
            """
            return new_position[g._X : g._Z+1] * g._DISTANCE_SCALE
        
    @jit        
    def _calculate_acceleration(self, body_index: int, position_data):
        """calculates the current acceleration for one body regarding the forces between the bodies

        Args:
            bodyIndex: index in body_data of the body, whose acceleration should be calculated
            
        Returns:
            np.array: vector of calculated new position
        """

        totalForce = 0
        pos_target_body = position_data[body_index]
        
        for curr_body_index, position in enumerate(position_data[:]):
            if curr_body_index == body_index:
                continue
            
            pos_curr_body = position
                
            force = g._G*((self.mass_data.get(body_index) * self.mass_data.get(curr_body_index))/(np.linalg.norm(pos_curr_body - pos_target_body))**3)*(pos_curr_body - pos_target_body)
            totalForce = totalForce + force
                
        acceleration = totalForce / self.mass_data.get(body_index)
        return acceleration   

    def _position_worker(self, p_in_queue, p_result_queue):
        """waits for tasks and calulates for each one the new position, new rendered position
            and velocity of one planet and puts it into the result queue.
            The result contains: body_index, new_position, rendered_position, new_velocity
        
        Args:
            p_in_queue: common job queue containing the tasks
            p_result_queue: common result_queue to put in the results of each process
        
        """   
        while True:           
            task = p_in_queue.get()
            if not self.mass_initialized:
               self.mass_data = self.m.get_mass_data()
               self.mass_initialized = True
            
            position_data = task[_POSITION_INDEX].copy()
            body_index = task[_PLANET_INDEX]
            delta_t = task[_DELTA_T_INDEX]
            
            curr_pos = position_data[body_index]
            curr_velocity = task[_VELOCITY_INDEX]
            
            accel = self._calculate_acceleration(body_index, position_data)
            new_pos = curr_pos +  curr_velocity * delta_t + 1/2 * accel * delta_t**2
            rendered_pos = self._calculate_rendering_position(new_pos)
            
            new_velocity = curr_velocity + accel * delta_t
            
            result = (body_index, new_pos, rendered_pos, new_velocity)
            p_result_queue.put(result)
            p_in_queue.task_done()
            print("Task done.")
            
    def _start_workers(self):
        """creates processes according to cpu_count and starts the worker function
            with the given queues from galaxymanager
        
        Args:
            m: Galaxymanager that manages all the queues
        """
        job_queue, result_queue = self.m.get_job_queue(), self.m.get_result_queue()
        nr_of_processes = cpu_count()
        print("starting worker, process count: ", nr_of_processes)
        processes = [Process(target = self._position_worker,
                args = (job_queue, result_queue))
            for i in range(nr_of_processes)]
        print("waiting for task...")
        for p in processes:
            p.start()

if __name__ == '__main__':
    from sys import argv, exit
    if len(argv) < 3:
        print('usage:', argv[0], 'server_IP server_socket')
        exit(0)
        
    worker = GalaxyWorker(argv[1], int(argv[2]))
    