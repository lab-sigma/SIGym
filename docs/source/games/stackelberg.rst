Stackelberg Game
=================================
.. image:: ../../_static/videos/stackelberg.gif
   :alt: game
   :width: 500
   :align: center

The user controls the leader, represented as the row player in the game. In each round, the leader selects a strategy, followed by the follower's response based on a user-specified behavior mode. The noisy utility feedback obtained by the leader/user when the leader chooses action i and the follower responds with action j is represented by the highlighted cell at coordinate (i, j). A lighter color indicates a higher utility for the user.

The concept of regret is defined as the difference between the Stackelberg leader utility (the maximum utility achievable in a single round if the leader has complete knowledge of the follower's private type) and the actual utility obtained by the user.

.. code-block:: python
   
   import numpy as np
   from sigym import Platform

   T = 10
   m, n = 3, 3
   trial = 5

   for behavior_mode in ["random", "best_response", "quantal_response", "mwu", "ftl", 'delta_suboptimal']:
      print("--------------------"*5, "Attacker mode: {}".format(attacker_mode), "--------------------"*5)
      rgt, cur_utility = 0.0, 0.0
      for tr in range(trial):
         
         # Initialize an instance
         agent, env = Platform(m, n, behavior_mode)
         u_sse = env.compute_SSE()

         x = [np.random.rand() for i in range(m)]
         temp = sum(x)
         x = [i/temp for i in x]

         for t in range(T):
         i_t, j_t = env.step(x, agent)
         cur_utility += env.compute_utility(i_t, j_t)
         
         # User-defined update rule - random update as an example
         x = [np.random.rand() for i in range(m)]
         temp = sum(x)
         x = [i/temp for i in x]

         rgt += (u_sse*T - cur_utility)/T
      print("The averaged regret you get over {} trials is {}".format(trial, rgt/trial))