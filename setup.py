#!/usr/bin/env python

from setuptools import setup, find_packages
import versioneer

setup(name='hiwenet',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Histogram-weighted Networks for Feature Extraction and Advance Analysis in Neuroscience',
      long_description='Histogram-weighted Networks for Feature Extraction and Advance Analysis in Neuroscience; hiwenet',
      author='Pradeep Reddy Raamana',
      author_email='raamana@gmail.com',
      url='https://github.com/raamana/hiwenet',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]), # ['neuropredict'],
      install_requires=['numpy', 'pyradigm', 'nibabel', 'networkx', 'medpy'],
      classifiers=[
              'Intended Audience :: Science/Research',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
              'Programming Language :: Python :: 3.5',
              'Programming Language :: Python :: 3.6',              
              'Programming Language :: Python :: 2.7',
          ],
      entry_points={
          "console_scripts": [
              "hiwenet=hiwenet.__main__:main",
          ]
      }

     )
