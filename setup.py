from setuptools import setup, find_packages

setup(
    name='sml',
    version='0.1',
    packages=find_packages(where='src'),
    install_requires=['torch', 'sdf', 'pandas'],
    package_dir={'': 'src'}
)
