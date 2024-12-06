from multiprocessing.managers import BaseManager
from multiprocessing import JoinableQueue, Queue
import socket
import sys
"""
The Galaxy manager is a class that needs to be run 
to enable the distribution to multiple devices. Needs
to be startet before anything else, because it registers
the queues and starts the connection server.
"""
class GalaxyManager(BaseManager):
        pass
        
if __name__ == '__main__':    
    master_port = 34002
    task_queue = JoinableQueue()
    result_queue = Queue()
    mass_data = dict()
    if len(sys.argv) < 3:
        print('usage:', sys.argv[0], 'server_IP server_socket')
        exit(0)
    server_ip = sys.argv[1]
    server_socket = int(sys.argv[2])
    GalaxyManager.register('get_job_queue', 
                         callable = lambda:task_queue)
    GalaxyManager.register('get_result_queue', 
                         callable = lambda:result_queue)
    GalaxyManager.register('get_mass_data', 
                         callable = lambda:mass_data)
    m = GalaxyManager(address = (server_ip, master_port), 
                    authkey = b'secret')
    print('starting queue server, socket', m.address)
    m.get_server().serve_forever()
    
    