from setuptools import setup
from Cython.Build import cythonize

setup(
    requires=["tensorflow", "numpy"],
    ext_modules=cythonize("computation.pyx")
)
