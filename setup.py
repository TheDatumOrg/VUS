from setuptools import setup, find_packages

__version__ = "1.0.1"

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setup(
    name="vus",
    version=__version__,
    description="Volume Under the Surface",
    classifiers=CLASSIFIERS,
    author="Teja",
    author_email="tejabogireddy19@gmail.com",
    packages=find_packages(),
    zip_safe=True,
    license="",
    url="https://github.com/bogireddytejareddy/VUS",
    entry_points={},
    install_requires=[
        "arch==5.3.1",
        "hurst==0.0.5",
        "matplotlib==3.5.3",
        "numpy==1.21.6",
        "pandas==1.3.5",
        "scikit-learn==0.22",
        "scipy==1.7.3",
        "statsmodels==0.13.2",
        "tsfresh==0.8.1"
        ]
)
