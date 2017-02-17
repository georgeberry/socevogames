import networkx as nx

from social_replicator import StageGame, SocialGroup

"""
Challenge here is to generate a completely connected graph where the interaction
pattern accords to one given by a SocialGroup.

Let's say there are 1000 nodes per group (evenly sized groups).
Then we can generate an interaction pattern from edge stubs and simulate a game.

What does this simulation look like? Go through everyone until there's
equilibrium? How do we know equilibrium will be reached?

We can get an analytical approximation following existing research. Is that
better than doing a sim?
"""
