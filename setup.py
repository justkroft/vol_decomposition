from setuptools import setup, Extension
import numpy

# Define the extension module
array_ops_module = Extension(
    'src.vol_decomposition',
    sources=['src/_c_src/vol_decomposition.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3']  # Optimization flag
)

setup(
    name='array_ops',
    version='1.0',
    description='Python C extension for array operations',
    ext_modules=[array_ops_module],
    packages=["src"],
    install_requires=['numpy']
)