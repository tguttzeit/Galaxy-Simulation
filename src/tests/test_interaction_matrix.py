""" Tests for module interaction_matrix"""
import pytest

import numpy as np

from OpenGL import GLUT
from OpenGL import GL

from vis_3d import interaction_matrix


@pytest.fixture
def matrix():
    if GLUT.glutGet(GLUT.GLUT_INIT_STATE) != 1:
        GLUT.glutInit([__name__])
    w = GLUT.glutCreateWindow(str.encode("pytest"))
    yield interaction_matrix.InteractionMatrix()
    GLUT.glutDestroyWindow(w)


def test_initial_matrix(matrix):
    m = matrix.get_current_matrix()
    expectation = np.identity(4)
    assert np.allclose(m, expectation, 1e-3)


def test_add_translation(matrix):
    t = [1, 2, 3]
    matrix.add_translation(*t)
    m = matrix.get_current_matrix().T
    expectation = np.identity(4)
    expectation[:3, 3] = t[:]
    assert np.allclose(m, expectation, 1e-3)
