from distutils.core import setup
from setuptools import find_packages

setup(
    name='EasyMultiPose',
    version='1.0.0',
    description='EasyMultiPose',
    packages=find_packages(),
    install_requires=[
        'numpy', 'torch', 'yaml'
    ]
)
