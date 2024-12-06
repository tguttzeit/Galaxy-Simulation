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

import time
import numpy as np
from galaxy_opt import Galaxy
from scalene import scalene_profiler

_DELTA_T = 30000
_NR_OF_PLANETS = 100
_STEP_LIMIT = 10
_WITH_SCALENE = True


class GalaxyOptStarter():
    """
    Widget with two buttons
    """

    def __init__(self):
        print("Initializing...")
        self.config_nr_of_planets = _NR_OF_PLANETS
        self.delta_t = _DELTA_T
        self.nr_of_entries = _NR_OF_PLANETS
        self.start_simulation()
    
    def start_simulation(self):
        """
        Start simulation and render processes.
        """
        if _WITH_SCALENE:
            scalene_profiler.start()
        self.setup()
        if _WITH_SCALENE:
            scalene_profiler.stop()
    
    def setup(self):            
        # Content
        # format: (pos_x, pos_y, pos_z, radius, c_red, c_green, c_blue, c_alpha)
        self.planet_array = np.zeros((self.nr_of_entries, 8), dtype=np.float32)
        self.planetary_system = Galaxy(self.nr_of_entries, True)
        self.planetary_system._startup(self.planet_array, self.delta_t, _STEP_LIMIT)

if __name__ == "__main__":
    simulation_gui = GalaxyOptStarter()
