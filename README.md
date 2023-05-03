# SIGym

*TODO: Expand this as the bounds of the project are defined; currently in stealth mode :)*

Strategic Interation Gym is a Python library for developing and exploring different algorithms for strategically interacting in game theoretic settings.
It aims to provide implementations of a variety of common environments and response mechanisms as well as an API for setting up custom environments and
algorithms.

To install the package, run `pip install -i https://test.pypi.org/simple/ SIGym-MinbiaoHan`.


Ported utilities/algorithmic implementations from here: https://github.com/lab-sigma/dynamic-stackelberg

You will need Gurobi (https://www.gurobi.com/) to run the code. Licenses can be obtained for free for academic purposes.

# Follower
We provide a number of attacker models for users to test their designed strategy in a Stackelberg game.

Random: arbitrarily returns a decision in each round.

Best: returns the optimal decision under a perfect Stackelberg equilibrium.

Quantal: under a quantal response equilibrium

MWU: returns the optimal decision using multiplicative weights update.

# Platform
The Platform class provides interfaces that help users evaluate the performance of their strategy. We provide three approaches of metrics:
SSE 
BSE
RME
Platform.step(): return the response for both sides
