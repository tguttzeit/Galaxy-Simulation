"""
Helper class for mouse interaction
"""
# helper class for mouse interaction
#
# Copyright (C) 2007  "Peter Roesch" <Peter.Roesch@fh-augsburg.de>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANtrans_y; without even the implied warrantrans_y of
# MERCHANTABILItrans_y or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
# or open http://www.fsf.org/licensing/licenses/gpl.html


from OpenGL import GLUT
from OpenGL import GL

from vis_3d.interaction_matrix import InteractionMatrix


class MouseInteractor:
    """Connection between mouse motion and transformation matrix"""

    def __init__(
        self, translation_scale: float = 0.1, rotation_scale: float = 0.2
    ) -> None:
        self.scaling_factor_rotation = rotation_scale
        self.scaling_factor_translation = translation_scale
        self.rotation_matrix = InteractionMatrix()
        self.translation_matrix = InteractionMatrix()
        self.mouse_button_pressed = None
        self.old_mouse_pos = [0, 0]

    def mouse_button(
        self, button: int, mode: int, x_pos: int, y_pos: int
    ) -> None:
        """Callback function for mouse button."""
        if mode == GLUT.GLUT_DOWN:
            self.mouse_button_pressed = button
        else:
            self.mouse_button_pressed = None
        self.old_mouse_pos[0], self.old_mouse_pos[1] = x_pos, y_pos
        GL.glClear(GL.GL_ACCUM_BUFFER_BIT)
        GLUT.glutPostRedisplay()

    def mouse_motion(self, x_pos: int, y_pos: int):
        """Callback function for mouse motion.

        Depending on the button pressed, the displacement of the
        mouse pointer is either converted into a translation vector
        or a rotation matrix."""

        delta_x = x_pos - self.old_mouse_pos[0]
        delta_y = y_pos - self.old_mouse_pos[1]
        if self.mouse_button_pressed == GLUT.GLUT_RIGHT_BUTTON:
            trans_x = delta_x * self.scaling_factor_translation
            trans_y = delta_y * self.scaling_factor_translation
            self.translation_matrix.add_translation(trans_x, -trans_y, 0)
        elif self.mouse_button_pressed == GLUT.GLUT_LEFT_BUTTON:
            rot_y = delta_x * self.scaling_factor_rotation
            self.rotation_matrix.add_rotation(rot_y, 0, 1, 0)
            rot_x = delta_y * self.scaling_factor_rotation
            self.rotation_matrix.add_rotation(rot_x, 1, 0, 0)
        else:
            trans_z = delta_y * self.scaling_factor_translation
            self.translation_matrix.add_translation(0, 0, trans_z)
        self.old_mouse_pos[0], self.old_mouse_pos[1] = x_pos, y_pos
        GL.glClear(GL.GL_ACCUM_BUFFER_BIT)
        GLUT.glutPostRedisplay()

    def apply_transformation(self) -> None:
        """Concatenation of the current translation and rotation
        matrices with the current OpenGL transformation matrix"""

        GL.glMultMatrixf(self.translation_matrix.get_current_matrix())
        GL.glMultMatrixf(self.rotation_matrix.get_current_matrix())

    def register_callbacks(self) -> None:
        """Initialise glut callback functions."""
        GLUT.glutMouseFunc(self.mouse_button)
        GLUT.glutMotionFunc(self.mouse_motion)
