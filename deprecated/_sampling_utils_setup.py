from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize([Extension("_sampling_utils_cdef", ["_sampling_utils_cdef.pyx"])]),
    include_dirs=[numpy.get_include()]
)
