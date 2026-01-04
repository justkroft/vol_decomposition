# python setup.py build_ext --inplace
import numpy
from setuptools import Extension, find_packages, setup

vol_decomposition_module = Extension(
    'src._vol_decomposition_c',
    sources=['src/_c_src/vol_decomposition.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3']
)

setup(
    name='vol_decomposition',
    version='1.0',
    description='Python C extension for volatility decomposition',
    packages=find_packages(),
    ext_modules=[vol_decomposition_module],
    install_requires=['numpy']
)
