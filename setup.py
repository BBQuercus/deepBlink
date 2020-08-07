"""Setup file for pypi package called deepblink."""

import io
import re
from os.path import dirname
from os.path import join

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as f:
        return f.read()


setup(
    # Description
    name="deepblink",
    version="0.0.4",
    license="MIT",
    description="Threshold independent detection and localization of diffraction-limited spots.",
    long_description="%s\n%s"
    % (
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S)
        .sub("", read("README.rst"))
        .replace(
            '.. raw:: html\n\n    <img src="https://github.com/bbquercus/deepblink/raw/master/images/logo.jpg" width="200px" align="right" alt="Logo of deepBlink.">',
            ".. image:: https://github.com/bbquercus/deepblink/raw/master/images/logo.jpg\n    :width: 200px\n    :align: right\n    :alt: Logo of deepBlink.",
        )
        .replace(
            """.. raw:: html

    <table width="100%">
      <tr>
        <th>Usage</th>
        <th>Example</th>
      </tr>
      <tr>
        <th min-width="200px" width="50%"><img src="https://github.com/bbquercus/deepblink/raw/master/images/usage.jpg" alt="Basic usage example of deepBlink."></th>
        <th min-width="200px" width="50%"><img src="https://github.com/bbquercus/deepblink/raw/master/images/example.jpg" alt="Example images processed with deepBlink."></th>
      </tr>
    </table>""",
            ".. image:: https://github.com/bbquercus/deepblink/raw/master/images/usage.jpg\n    :width: 100%\n    :alt: Basic usage example of deepBlink.\n\n.. image:: https://github.com/bbquercus/deepblink/raw/master/images/example.jpg\n    :width: 100%\n    :alt: Example images processed with deepBlink.",
        ),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("docs/changelog.rst")),
    ),
    # Installation
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.6, !=3.9.*",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "pillow",
        "scikit-image",
        "tensorflow>=2.0",
        "wandb>=0.7.0"
        # "scipy",
    ],
    entry_points={"console_scripts": ["deepblink = deepblink.cli:main"]},
    # Metadata
    author="Bastian Eichenberger, YinXiu Zhan",
    author_email="bastian@eichenbergers.ch, yinxiuzhan89@gmail.com",
    url="https://github.com/bbquercus/deepblink/",
    project_urls={
        "Documentation": "https://deepblink.readthedocs.io/",
        "Changelog": "https://deepblink.readthedocs.io/en/latest/changelog.html",
        "Issue Tracker": "https://github.com/bbquercus/deepblink/issues",
    },
    keywords=["deep-learning", "biomedical", "image analysis", "spot detection"],
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Life",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Utilities",
    ],
)
