"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
import subprocess
import sys
from codecs import open
from pathlib import Path

from setuptools import setup, find_packages

here = Path(__file__).absolute().parent

error = subprocess.call(['make', 'meta'])
if error:
    print(f"failed to run 'make meta' with error: {error}")
    sys.exit(-1)

with open(here / Path('README.md'), encoding='utf-8') as f:
    long_description = f.read().replace('\r\n', '\n')

_version = {}
with open(here / Path('reinforcement/_version.py', mode='r')) as f:
    exec(f.read(), _version)

setup(
    name='reinforcement',
    version=_version['__version__'],
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
    extras_require={
        'tf_gpu': ['numpy<1.17', 'tensorflow-gpu==1.14'],
        'tf_cpu': ['numpy<1.17', 'tensorflow==1.14'],
        'test': ['numpy<1.17', 'tensorflow==1.14', 'pytest', 'pytest-rerunfailures', 'matplotlib', 'gym'],
    },
    project_urls={
        'Bug Reports': 'https://github.com/SwamyDev/reinforcement/issues',
        'Source': 'https://github.com/SwamyDev/reinforcement/',
    },
)
