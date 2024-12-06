"""
OpenGL output for gravity simulation
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
import sys
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from OpenGL import GLUT
from OpenGL import GL
from OpenGL import GLU

from .mouse_interactor import MouseInteractor

# initial window parameters
_WINDOW_SIZE = (512, 512)
_WINDOW_POSITION = (100, 100)
_LIGHT_POSITION = (2, 2, 3)
_CAMERA_POSITION = (0, 0, 2)


class GalaxyRenderer:
    """
    Class containing OpenGL code
    """

    def __init__(
        self,
        bodies_shm: SharedMemory,
        flags_shm: SharedMemory,
        fadeout_fraction: float,
    ) -> None:
        self.flags_shm = flags_shm
        self.flags = flags_shm.buf
        self.bodies_shm = bodies_shm
        self.fadeout_fraction = fadeout_fraction
        float32_nr = len(bodies_shm.buf) // 4
        shared_bodies = np.ndarray(
            shape=(float32_nr,), dtype=np.float32, buffer=bodies_shm.buf
        )
        self.bodies = shared_bodies.reshape(-1, 8)
        self.sphere = None
        self.init_glut()
        self.init_gl()
        self.mouse_interactor = MouseInteractor(0.01, 1)
        self.mouse_interactor.register_callbacks()

    def init_glut(self):
        """
        Set up window and main callback functions
        """
        GLUT.glutInit(["Galaxy Renderer"])
        GLUT.glutInitDisplayMode(
            GLUT.GLUT_DOUBLE
            | GLUT.GLUT_RGB
            | GLUT.GLUT_DEPTH
            | GLUT.GLUT_ACCUM
        )
        GLUT.glutInitWindowSize(_WINDOW_SIZE[0], _WINDOW_SIZE[1])
        GLUT.glutInitWindowPosition(_WINDOW_POSITION[0], _WINDOW_POSITION[1])
        GLUT.glutCreateWindow(str.encode("Galaxy Renderer"))
        GLUT.glutDisplayFunc(self.render)
        GLUT.glutIdleFunc(self.update_positions)

    def init_gl(self):
        """
        Initialise OpenGL settings
        """
        self.sphere = GL.glGenLists(1)
        GL.glNewList(self.sphere, GL.GL_COMPILE)
        quad_obj = GLU.gluNewQuadric()
        GLU.gluQuadricDrawStyle(quad_obj, GLU.GLU_FILL)
        GLU.gluQuadricNormals(quad_obj, GLU.GLU_SMOOTH)
        GLU.gluSphere(quad_obj, 1, 16, 16)
        GL.glEndList()
        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_LIGHTING)
        # make sure normal vectors of scaled spheres are normalised
        GL.glEnable(GL.GL_NORMALIZE)
        GL.glEnable(GL.GL_LIGHT0)
        light_pos = list(_LIGHT_POSITION) + [1]
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, light_pos)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT, [0.7, 0.7, 0.7, 1])
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, [0.7, 0.7, 0.7, 1])
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_SPECULAR, [0.1, 0.1, 0.1, 1])
        GL.glMaterialf(GL.GL_FRONT, GL.GL_SHININESS, 20)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(60, 1, 0.01, 10)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glClear(
            GL.GL_COLOR_BUFFER_BIT
            | GL.GL_DEPTH_BUFFER_BIT
            | GL.GL_ACCUM_BUFFER_BIT
        )

    def render(self):
        """
        Render the scene using the sphere display list
        """
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        x_size = GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH)
        y_size = GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT)
        GLU.gluPerspective(60, float(x_size) / float(y_size), 0.05, 10)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glTranslatef(
            -_CAMERA_POSITION[0], -_CAMERA_POSITION[1], -_CAMERA_POSITION[2]
        )
        self.mouse_interactor.apply_transformation()
        # load accumulation buffer as background
        GL.glAccum(GL.GL_RETURN, 1.0)
        for body in self.bodies[:]:
            GL.glPushMatrix()
            GL.glTranslatef(body[0], body[1], body[2])
            GL.glScalef(body[3], body[3], body[3])
            GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE, body[4:])
            GL.glCallList(self.sphere)
            GL.glPopMatrix()
        # store faded frame buffer to accumulation buffer
        GL.glAccum(GL.GL_LOAD, self.fadeout_fraction)
        GLUT.glutSwapBuffers()

    @staticmethod
    def start():
        """
        Start the GLUT event loop.
        """
        GLUT.glutMainLoop()

    def update_positions(self):
        """
        Check if new positions are present
        """
        if self.flags[1] != 0:
            print("renderer exiting ...")
            for share in (self.flags_shm, self.bodies_shm):
                share.close()
            GLUT.glutLeaveMainLoop()
        # check if redraw is required
        elif self.flags[0] != 0:
            self.flags[0] = 0
            GLUT.glutPostRedisplay()
        else:
            time.sleep(1 / 250)


def stop():
    """
    Close application
    """

    sys.exit(0)


def startup(
    shared_bodies_name: str,
    shared_flags_name: str,
    fadeout_fraction: float = 0.7,
) -> None:
    """
    Create GalaxyRenderer instance and start rendering

    Args:
        shared_list_name: name of shared memory list
    """
    bodies_shm = SharedMemory(shared_bodies_name)
    flags_shm = SharedMemory(shared_flags_name)
    print("creating renderer")
    renderer = GalaxyRenderer(bodies_shm, flags_shm, fadeout_fraction)
    print("starting renderer")
    renderer.start()
