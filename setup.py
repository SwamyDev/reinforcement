"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from codecs import open
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read().replace('\r\n', '\n')

setup(
    name='reinforcement',
    version='1.0.5',
    description='A reinforcement learning module',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SwamyDev/reinforcement',
    author='Bernhard Raml',
    author_email='pypi-reinforcment@googlegroups.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='AI reinforcement learning',
    packages=find_packages(exclude=['build', 'dist', 'reinforcement.egg-info', 'test_integration', 'tests']),
    install_requires=['numpy', 'tensorflow'],
    extras_require={
        'test': ['pytest', 'coverage'],
    },
    project_urls={
        'Bug Reports': 'https://github.com/SwamyDev/reinforcement/issues',
        'Source': 'https://github.com/SwamyDev/reinforcement/',
    },
)