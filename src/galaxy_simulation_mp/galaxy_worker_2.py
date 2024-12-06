from multiprocessing import Process, cpu_count, Pool
from multiprocessing import JoinableQueue, Queue

from galaxy_manager import GalaxyManager
import galaxy as g

import numpy as np

_PLANET_INDEX = 0
_BODY_DATA_INDEX = 1
_DELTA_T_INDEX = 2

def calculate_rendering_position(new_position) -> None:
        """scales the new positions that should be renderd using the current positions from body_data
            positions can only be rendered between -1 and 1, so they need to be scaled
        """
        return new_position[g._POS_X_DATA : g._POS_Z_DATA+1] * g._DISTANCE_SCALE

def _calc_new_position(accel, curr_body, delta_t):
    curr_pos = curr_body[g._POS_X_DATA:g._POS_Z_DATA+1].copy()
    curr_velocity = curr_body[g._VELOCITY_X_DATA:g._VELOCITY_Z_DATA+1].copy()
    
    new_pos = curr_pos +  curr_velocity * delta_t + 1/2 * accel * delta_t**2
    rendered_pos = calculate_rendering_position(new_pos)
        
    new_velocity = curr_velocity + accel * delta_t
    return (new_pos, rendered_pos, new_velocity)

def _calculate_force(central_body, reference_body):
        """
        """
        pos_central_body = central_body[g._POS_X_DATA:g._POS_Z_DATA+1].copy()
        pos_curr_body = reference_body[g._POS_X_DATA:g._POS_Z_DATA+1].copy()
        force = g._G*((central_body[g._MASS_DATA] * reference_body[g._MASS_DATA])/(np.linalg.norm(pos_curr_body - pos_central_body))**3)*(pos_curr_body - pos_central_body)
        print(f"calulated force: {force}, of central_body {central_body} and reference_body {reference_body} \n")
        return force

def _force_worker(p_in_queue, p_result_queue):
    while True:
        task = p_in_queue.get()
        print("force worker task: ", task)
        central_body = task[0]
        reference_body = task[1]
        result = _calculate_force(central_body, reference_body)
        p_result_queue.put(result)
        p_in_queue.task_done()

def prepare_in_queue(body_index, body_data):
    p_in_queue = JoinableQueue()
    for i in range(len(body_data)):
        if not i == body_index:
            p_in_queue.put((body_data[body_index], body_data[i]))
    return p_in_queue

def _prep_processes(m):
    job_queue, result_queue = m.get_job_queue(), m.get_result_queue()
    while True:
        print("waiting for tasks...")
        task = job_queue.get()

        body_data = task[_BODY_DATA_INDEX].copy()
        body_index = task[_PLANET_INDEX]
        delta_t = task[_DELTA_T_INDEX]
        curr_body = body_data[body_index]  

        p_in_queue = prepare_in_queue(body_index, body_data)
        p_result_queue = Queue()
        print("prepared in queue: ", p_in_queue.qsize())
        processes = [Process(target = _force_worker,
                args = (p_in_queue, p_result_queue))
            for i in range(len(body_data))]
        for p in processes:
            p.start()

        p_in_queue.join()
        print("queue joined ", p_result_queue.qsize())
        total_force = np.zeros(shape=3)
        while not p_result_queue.empty():
            total_force = total_force + p_result_queue.get()
        print("read result of total_force ", total_force)
        accel = total_force / curr_body[g._MASS_DATA]
        result = (body_index, ) + _calc_new_position(accel, curr_body, delta_t)

        result_queue.put(result)
        job_queue.task_done()
         

if __name__ == '__main__':
    from sys import argv, exit
    if len(argv) < 3:
        print('usage:', argv[0], 'server_IP server_socket')
        exit(0)
    server_ip = argv[1]
    server_socket = int(argv[2])
    GalaxyManager.register('get_job_queue')
    GalaxyManager.register('get_result_queue')
    m = GalaxyManager(address=(server_ip, server_socket), authkey = b'secret')
    m.connect()
    print("sucessfully connected", m.address)
    _prep_processes(m)