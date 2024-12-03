#! /usr/bin/env python
"""A template."""

import codecs
import os

from setuptools import find_packages, setup
from torch_concepts import __version__

DISTNAME = 'pytorch_concepts'
DESCRIPTION = ' Concept-Based Deep Learning Library for PyTorch.'
with codecs.open('README.md', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'PyC Team'
MAINTAINER_EMAIL = 'barbiero@tutanota.com'
URL = 'https://github.com/pyc-team/pytorch_concepts'
LICENSE = 'Apache 2.0'
DOWNLOAD_URL = 'https://github.com/pyc-team/pytorch_concepts'
VERSION = __version__
INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'scikit-learn',
    'pandas',
    'torch',
    'opencv-python',
]
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
