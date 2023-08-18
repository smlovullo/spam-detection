from setuptools import setup, find_packages

setup(
    name='spam-detection',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires=">=3.9",
    install_requires=[''],
    test_suite='tests',
)