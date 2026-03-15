# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='lambwaves',
    version='0.0.0',
    description='Tools to calculate and plot Lamb wave dispersion curves',
    long_description=readme,
    license=license,
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib']
)
