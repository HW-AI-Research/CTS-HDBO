import re
from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


setup(name='cts',
      packages=[package for package in find_packages()
                if package.startswith('cts')],
      install_requires=[
          'pyyaml',
          'botorch==0.8.1',
          "numpy>=1.21",
          "torch>=1.3",
          "lasso-bench-fork-leoiv==0.0.6",
        ],
      extras_require=None,
      description='Cylindrical Thompson Sampling for High-Dimensional Bayesian Optimization',
      author='Kerrick Johnstonbaugh',
      url='https://github.com/BahadorRashidi/CTS-BO/tree/main',
      author_email='kjohnstonbaugh97@gmail.com',
      version='1.0.0')