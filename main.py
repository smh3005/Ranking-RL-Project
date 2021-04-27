from UCB import UCB
from ELO import ELO

# Run Tests
ucb = UCB()
ucb.run_experiment(n_teams=5,
                   n_judges=1,
                   true_q=[0.1,0.3,0.5,0.7,0.9],
                   n_episodes=200,
                   c=0.1,
                   var=0.001
                  )

elo = ELO()
elo.run_experiment(n_teams=5,
                   n_judges=1,
                   true_q=[0.1,0.3,0.5,0.7,0.9],
                   n_episodes=200,
                   c=0.1,
                   var=0.001
                  )