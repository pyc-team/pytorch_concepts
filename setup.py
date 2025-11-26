#! /usr/bin/env python

"""A template."""

import codecs
from pathlib import Path

from setuptools import find_packages, setup

about = {}
version_file = Path(__file__).parent / "torch_concepts" / "_version.py"
exec(version_file.read_text(), about)

DISTNAME = 'pytorch_concepts'
DESCRIPTION = ' Concept-Based Deep Learning Library for PyTorch.'
with codecs.open('README.md', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()

MAINTAINER = 'PyC Team'
MAINTAINER_EMAIL = 'pyc.devteam@gmail.com'
URL = 'https://github.com/pyc-team/pytorch_concepts'
LICENSE = 'Apache 2.0'
DOWNLOAD_URL = 'https://github.com/pyc-team/pytorch_concepts'
VERSION = about["__version__"]
INSTALL_REQUIRES = [
    'scikit-learn',
    'scipy',
    'torch',
    'pytorch-minimize',
    'pytorch-lightning',
    'networkx',
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
    'data': [
        'opencv-python',
        'pandas',
        'torchvision',
        'pgmpy',
        'bnlearn',
        'datasets',
        'transformers',
        'tables',
    ],
    'tests': [
        'pytest-cov',
        'pytest',
    ],
    'docs': [
        'matplotlib',
        'numpydoc',
        'furo',
        'sphinx-gallery',
        'sphinx',
        'sphinx_design',
        'sphinxext-opengraph',
        'sphinx-copybutton',
        'myst-nb',
        'sphinx-hoverxref',
    ],
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(include=["torch_concepts", "torch_concepts.*"]),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    use_scm_version=True,  # Adding so that package data is respected
    setup_requires=["setuptools_scm"],
    include_package_data=True,  # We need this to include static assets (images)
    package_data={
        "torch_concepts": ["assets/*"],
    },
)
