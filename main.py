import numpy as np

from epsilongreedy import EpsilonGreedy
from ucb import UCB
from ELO import ELO
true_q_15 = np.linspace(0.05, 0.95, 15)

ucb3 = UCB()
ucb3.run_experiment(n_teams=15,
                    n_judges=1,
                    true_q=true_q_15,
                    n_episodes=200,
                    c=0.7,
                    var=0.001,
                    top_n=5
                    )

egreedy3 = EpsilonGreedy()
egreedy3.run_experiment(n_teams=15,
                        n_judges=1,
                        true_q=true_q_15,
                        n_episodes=200,
                        c=0.1,
                        var=0.001,
                        top_n=5
                        )

elo = ELO()
elo.run_experiment(n_teams=15,
                   n_judges=1,
                   true_q=true_q_15,
                   n_episodes=200,
                   c=0.1,
                   var=0.001,
                   top_n=5
                  )

import matplotlib.pyplot as plt
plt.legend()
plt.show()