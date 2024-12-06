"""
setup file for project galaxy_renderer
"""

import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="vis_3d",
    version="0.1.1",
    # replace underline characters by spaces
    author="Peter Rösch",
    author_email="Peter.Roesch@hs-augsburg.de",
    description=("Output spheres in 3D using OpenGL"),
    license="GPL",
    keywords="3D rendering",
    url="https://hs-augsburg.de",
    packages=["vis_3d"],
    long_description=read("README.md"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Demonstration",
        "License :: GPL License",
    ],
    python_requires=">=3.10, <4",
    install_requires=["numpy>=1.21", "PyOpenGL>=3.1"],
)
