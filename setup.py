"""Setup file for pypi package called deepblink."""

from setuptools import find_packages
from setuptools import setup

setup(
    # Description
    name="deepblink",
    version="0.0.4",
    license="MIT",
    description="Threshold independent detection and localization of diffraction-limited spots.",
    long_description="""In biomedical microscopy data, a common task involves the detection of diffraction-limited spots that visualize single proteins, domains, mRNAs, and many more. These spots were traditionally detected with mathematical operators such as Laplacian of Gaussian. These operators, however, rely on human input ranging from image-intensity thresholds, approximative spot sizes, etc. This process is tedious and not always reliable.

    DeepBlink relies on neural networks to automatically find spots without the need for human intervention. DeepBlink is available as a ready-to-use command-line interface.

    All deepBlink wheels distributed on PyPI are MIT licensed.
    """,
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
