#! /usr/bin/env python

"""A template."""

import codecs

from setuptools import find_packages, setup
from torch_concepts import __version__

DISTNAME = 'pytorch_concepts'
DESCRIPTION = ' Concept-Based Deep Learning Library for PyTorch.'
with codecs.open('README.md', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()

MAINTAINER = 'PyC Team'
MAINTAINER_EMAIL = 'pyc.devteam@gmail.com'
URL = 'https://github.com/pyc-team/pytorch_concepts'
LICENSE = 'Apache 2.0'
DOWNLOAD_URL = 'https://github.com/pyc-team/pytorch_concepts'
VERSION = __version__
INSTALL_REQUIRES = [
    'numpy',
    'opencv-python',
    'pandas',
    'scikit-learn',
    'scipy',
    'torch',
]
CLASSIFIERS = [
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved',
    'Operating System :: MacOS',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development',
]
EXTRAS_REQUIRE = {
    'tests': [
        'pytest-cov',
        'pytest',
    ],
    'docs': [
        'matplotlib',
        'numpydoc',
        'sphinx_rtd_theme',
        'sphinx-gallery',
        'sphinx',
    ],
}

setup(
    name=DISTNAME,
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
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,  # We need this to include static assets (images)
    package_data={
        "pytorch_concepts": ["data/traffic_construction/assets/*"],
    }
)
