from UCB_PREV1 import UCB_PREV1
from UCB_PREV2 import UCB_PREV2
from UCB_PREV3 import UCB_PREV3
from EGREEDY_PREV1 import EGREEDY_PREV1
from EGREEDY_PREV2 import EGREEDY_PREV2
from EGREEDY_PREV3 import EGREEDY_PREV3
from ELO import ELO
import numpy as np

true_q_15 = np.linspace(0.05, 0.95, 15)

# Run Tests
ucb1 = UCB_PREV1()
ucb1.run_experiment(n_teams=15,
                    n_judges=1,
                    true_q=true_q_15,
                    n_episodes=200,
                    c=0.7,
                    var=0.001,
                    top_n=5
                    )

ucb2 = UCB_PREV2()
ucb2.run_experiment(n_teams=15,
                    n_judges=1,
                    true_q=true_q_15,
                    n_episodes=200,
                    c=0.7,
                    var=0.001,
                    top_n=5
                    )

ucb3 = UCB_PREV3()
ucb3.run_experiment(n_teams=15,
                    n_judges=1,
                    true_q=true_q_15,
                    n_episodes=200,
                    c=0.7,
                    var=0.001,
                    top_n=5
                    )

egreedy1 = EGREEDY_PREV1()
egreedy1.run_experiment(n_teams=15,
                        n_judges=1,
                        true_q=true_q_15,
                        n_episodes=200,
                        c=0.1,
                        var=0.001,
                        top_n=5
                        )

egreedy2 = EGREEDY_PREV2()
egreedy2.run_experiment(n_teams=15,
                        n_judges=1,
                        true_q=true_q_15,
                        n_episodes=200,
                        c=0.1,
                        var=0.001,
                        top_n=5
                        )

egreedy3 = EGREEDY_PREV3()
egreedy3.run_experiment(n_teams=15,
                        n_judges=1,
                        true_q=true_q_15,
                        n_episodes=200,
                        c=0.1,
                        var=0.001,
                        top_n=5
                        )

# elo = ELO()
# elo.run_experiment(n_teams=5,
#                    n_judges=1,
#                    true_q=[0.1,0.3,0.5,0.7,0.9],
#                    n_episodes=200,
#                    c=0.1,
#                    var=0.001
#                   )
