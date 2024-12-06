""" simple PyQt5 simulation controller """
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

from pathlib import Path
import sys
import time
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QLineEdit, QLabel
from PyQt5.QtGui import QDoubleValidator
from vis_3d import renderer
from galaxy import Galaxy
import json

class SimulationGUI(QtWidgets.QWidget):
    """
    Widget with two buttons
    """

    def __init__(self, nr_of_entries, delta_t, server_ip, server_port):
        self.config_nr_of_planets = nr_of_entries
        self.delta_t = delta_t
        self.nr_of_entries = nr_of_entries
        self.nr_of_entries_changed = False
        self.server_ip = server_ip
        self.server_port = server_port
        # GUI 
        
        QtWidgets.QWidget.__init__(self)
        self.setGeometry(0, 0, 420, 125)
        self.setWindowTitle("Simulation")
        
        self.delta_t_label = QLabel(self)
        self.delta_t_label.setText("Delta t:")
        self.delta_t_label.setGeometry(60, 10, 50, 35)

        self.delta_t_textbox = QLineEdit(self)
        self.delta_t_textbox.setGeometry(110, 10, 70, 35)
        self.delta_t_textbox.setValidator(QDoubleValidator())
        self.delta_t_textbox.returnPressed.connect(self.use_custom_delta_t)
        
        self.nr_of_entries_label = QLabel(self)
        self.nr_of_entries_label.setText("Nr. of bodies:")
        self.nr_of_entries_label.setGeometry(210, 10, 100, 35)

        self.nr_of_entries_textbox = QLineEdit(self)
        self.nr_of_entries_textbox.setGeometry(300, 10, 70, 35)
        self.nr_of_entries_textbox.setValidator(QDoubleValidator())
        self.nr_of_entries_textbox.returnPressed.connect(self.use_custom_nr_of_entries)

        self.start_button = QtWidgets.QPushButton("Start", self)
        self.start_button.setGeometry(30, 55, 100, 35)
        self.start_button.clicked.connect(self.start_simulation)
        
        self.quit_sim_button = QtWidgets.QPushButton("Quit Sim", self)
        self.quit_sim_button.setGeometry(160, 55, 100, 35)
        self.quit_sim_button.clicked.connect(self.exit_simulation)

        self.quit_button = QtWidgets.QPushButton("Quit", self)
        self.quit_button.setGeometry(290, 55, 100, 35)
        self.quit_button.clicked.connect(self.exit_application)
        
        self.status_label = QLabel(self)
        self.status_label.setText("...")
        self.status_label.setGeometry(10, 95, 380, 20)
        
    

    def use_custom_delta_t(self):
        """
        In case a new delta t was put in, it is used instead of the delta t from the
        config file
        """
        old_delta_t = self.delta_t
        self.delta_t = float(self.delta_t_textbox.text())
        self.status_label.setText(f"Delta t is changed to {self.delta_t} | Was: {old_delta_t}")
    
    def use_custom_nr_of_entries(self):
        """
        In case a number of bodies was put in, it is used instead of the number of planets from the
        config file
        """
        old_nr_of_entries = self.nr_of_entries
        self.nr_of_entries = int(self.nr_of_entries_textbox.text())
        if(self.nr_of_entries == 0):
            self.nr_of_entries = self.config_nr_of_planets
            self.status_label.setText(f"Number of bodies is changed to {self.nr_of_entries} | start solar system")
            self.nr_of_entries_changed = False # to start solar system sim
        else:
            self.status_label.setText(f"Number of bodies is changed to {self.nr_of_entries} | Was: {old_nr_of_entries}")
            self.nr_of_entries_changed = True # to start random simulation with black hole

    def start_simulation(self):
        """
        Start simulation and render processes.
        """
        self.setup()
        self.status_label.setText(f"Started simulation with a delta t: {self.delta_t} and nr of bodies: {self.nr_of_entries}")
        self.simulation_process.start()
        self.render_process.start()
    
    def setup(self):
        # add black hole, if nr of bodies is changed
        if(self.nr_of_entries_changed):
            self.nr_of_entries = self.nr_of_entries + 1
            
        # Content
        # format: (pos_x, pos_y, pos_z, radius, c_red, c_green, c_blue, c_alpha)
        self.planet_array = np.zeros((self.nr_of_entries, 8), dtype=np.float32)
        # format: [do_render, do_terminate]
        self.shared_flags = SharedMemory(create=True, size=2)
        self.shared_flags.buf[0] = 1
        self.shared_flags.buf[1] = 0
        self.shared_planets = SharedMemory(
            create=True, size=self.planet_array.nbytes
        )
        
        self.render_process = Process(
            target=renderer.startup,
            args=(self.shared_planets.name, self.shared_flags.name),
        )
        
        self.planetary_system = Galaxy(self.nr_of_entries, self.nr_of_entries_changed, self.server_ip, self.server_port)
        self.simulation_process = Process(
            target=self.planetary_system.startup,
            args=(self.shared_planets.name, self.shared_flags.name, self.delta_t)
        )

    def exit_simulation(self):
        """
        Stop simulation and resets application.
        """
        self.shared_flags.buf[1] = 1
        time.sleep(0.5)
        self.simulation_process.terminate()
        self.render_process.terminate()
        for s in (self.shared_flags, self.shared_planets):
            s.close()
            s.unlink()
        self.status_label.setText("application reseted")
    
    def exit_application(self):
        """
        Stop simulation and exit.
        """
        
        self.exit_simulation()
        self.close()
        

def _main(argv):
    """
    Main function to avoid pylint complains concerning constant names.
    """
    cfg_path = f"{Path(__file__, '..').resolve()}/galaxy_cfg.json"
    with open (cfg_path) as cfg_file:
        cfg = json.load(cfg_file)
    if len(sys.argv) < 3:
        print('usage:', sys.argv[0], 'server_IP server_socket')
        exit(0)
    server_ip = sys.argv[1]
    server_socket = int(sys.argv[2])
    nr_of_planets = len(cfg["planets"])
    delta_t = cfg["delta_t"]
    app = QtWidgets.QApplication(argv)
    simulation_gui = SimulationGUI(nr_of_planets, delta_t, server_ip, server_socket)
    simulation_gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    _main(sys.argv)
