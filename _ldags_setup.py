from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize([Extension("_ldags_train", ["_ldags_train.pyx"])]),
    include_dirs=[numpy.get_include()]
)
