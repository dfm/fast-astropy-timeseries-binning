import sys

try:
    from skbuild import setup
    import nanobind as nanobind
except ImportError:
    print(
        "The preferred way to invoke 'setup.py' is via pip, as in 'pip "
        "install .'. If you wish to run the setup script directly, you must "
        "first install the build dependencies listed in pyproject.toml!",
        file=sys.stderr,
    )
    raise

setup(
    name="fast-astropy-timeseries-binning",
    author="Dan Foreman-Mackey",
    author_email="dfm@dfm.io",
    description="Faster binning functions for astropy.timeseries",
    long_description="Faster binning functions for astropy.timeseries",
    long_description_content_type="text/markdown",
    url="https://github.com/dfm/fast-astropy-timeseries-binning",
    python_requires=">=3.8",
    license="Apache",
    packages=["fast_astropy_timeseries_binning"],
    package_dir={"": "src"},
    cmake_install_dir="src/fast_astropy_timeseries_binning",
    include_package_data=True,
)
