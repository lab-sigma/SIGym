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
   from src.SIGym.sigym import Follower, Platform
   from src.SIGym.General_Stackelberg.dynamic_stackelberg import dyse, SSE, utils


   T = 10
   m, n = 3, 3
   trial = 5

   for attacker_mode in ["random", "best_response", "quantal_response", "mwu", "ftl", 'delta_suboptimal']:
      print("--------------------"*5, "Attacker mode: {}".format(attacker_mode), "--------------------"*5)
      rgt, cur_utility = 0.0, 0.0
      for tr in range(trial):

         # generate a random game
         R, C, _ = utils.setup_random_game_int(m, n , ntypes=1)

         R, C = R[0], C[0]
         agent = Follower(utility_matrix=C, behavior_mode=attacker_mode)  # Instantiate a follower class using the utility matrix and the behavior model
         env = Platform(R, C) # Instantiate a platform class using the reward matrix and the utility matrix
         u_sse = env.compute_SSE()

         x = [np.random.rand() for i in range(m)]
         temp = sum(x)
         x = [i/temp for i in x]
         for t in range(T):
               i_t, j_t = env.step(x, agent, R) # the platform takes a step based on the leader strategy and the follower model
               cur_utility += R[i_t][j_t]
               x = [np.random.rand() for i in range(m)]
               temp = sum(x)
               x = [i/temp for i in x]

         rgt += (u_sse*T - cur_utility)/T
      print("The averaged regret you get over {} trials is {}".format(trial, rgt/trial))