from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='sigym',
    version='1.0',
    description='Implementations of various programs for studying strategic interaction in game theoretic settings.',
    license="MIT",
    long_description=long_description,
    author='Minbiao Han, Quinn Dawkins',
    author_email='mh2ye@virginia.edu, qed4wg@virginia.edu',
    url="https://github.com/lab-sigma/SIGym",
    packages=['sigym'],
    install_requires=['numpy', 'gurobipy', 'pandas'],
)
