from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='dynamic_stackelberg',
    version='1.0',
    description='Implementations of various programs for studying optimal behavior in multiround bayesian stackelberg games',
    license="MIT",
    long_description=long_description,
    author='Minbiao Han, Quinn Dawkins',
    author_email='mh2ye@virginia.edu, qed4wg@virginia.edu',
    url="https://github.com/lab-sigma/dynamic-stackelberg",
    packages=['dynamic_stackelberg'],
    install_requires=['numpy', 'gurobipy', 'pandas'],
)
