from setuptools import find_packages
from setuptools import setup
import shlex
import subprocess


def git_version():
    cmd = 'git log --format="%h" -n 1'
    return subprocess.check_output(shlex.split(cmd)).decode().strip()


version = git_version()

setup(
    name='hpe3d',
    version=version,
    packages=find_packages(),
    install_requires=[],  # see requirements.txt
    author='Dorian Henning',
    author_email='dorian.henning@gmail.com',
    license='MIT',
    url='https://github.com/dorianhenning/3dhpe',
    description='Human Pose Estimation in 3D',
)
