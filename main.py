import numpy as np

from epsilongreedy import EpsilonGreedy
from ucb import UCB
from ELO import ELO
from JudgingSimulator import JudgingSimulator
import matplotlib.pyplot as plt

true_q = np.linspace(0.05, 0.95, 15)
n_judges = 1
opinion_var = 0.001
eval_var = 0.001
sim = JudgingSimulator(true_q, n_judges, opinion_var, eval_var) 

ucb = UCB(sim, c=0.7)
ucb.run_experiment(n_episodes=20, top_n=5).print_results()
ucb.rbo(0.5)

egreedy = EpsilonGreedy(sim, epsilon=0.1)
egreedy.run_experiment(n_episodes=20, top_n=5).print_results().plot('inclusion')

elo = ELO(sim)
elo.run_experiment(n_episodes=20, top_n=5).print_results().plot('inclusion')

plt.ylim(-0.2, 1.2)
plt.legend()
plt.show()