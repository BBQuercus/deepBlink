"""Setup file for pypi package called deepblink."""

# TODO write makefile for basic deployment using
# python setup.py sdist
# twine upload dist/latest-version.tar.gz

import textwrap
from setuptools import find_packages
from setuptools import setup

setup(
    # Description
    name="deepblink",
    version="0.1.4",
    license="MIT",
    description="Threshold independent detection and localization of diffraction-limited spots.",
    long_description_content_type="text/plain",
    long_description=textwrap.dedent(
        """\
        In biomedical microscopy data, a common task involves the detection of diffraction-limited spots that
        visualize single proteins, domains, mRNAs, and many more. These spots were traditionally detected with
        mathematical operators such as Laplacian of Gaussian. These operators, however, rely on human input ranging
        from image-intensity thresholds, approximative spot sizes, etc. This process is tedious and not always
        reliable.\n
        DeepBlink relies on neural networks to automatically find spots without the need for human
        intervention. DeepBlink is available as a ready-to-use command-line interface.\n
        All deepBlink wheels distributed on PyPI are MIT licensed."""
    ),
    # Installation
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "pyyaml",
        "requests",
        "scikit-image",
        "scipy",
        "tensorflow>=2.0.0",
    ],
    entry_points={"console_scripts": ["deepblink = deepblink.cli:main"]},
    # Metadata
    author="Bastian Eichenberger, YinXiu Zhan",
    author_email="bastian@eichenbergers.ch, yinxiuzhan89@gmail.com",
    url="https://github.com/bbquercus/deepblink/",
    project_urls={
        "Documentation": "https://github.com/BBQuercus/deepBlink/wiki",
        "Changelog": "https://github.com/BBQuercus/deepBlink/releases",
        "Issue Tracker": "https://github.com/bbquercus/deepblink/issues",
    },
    keywords=["deep-learning", "biomedical", "image analysis", "spot detection"],
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Life",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Utilities",
    ],
)
