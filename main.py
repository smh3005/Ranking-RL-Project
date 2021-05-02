import numpy as np

from epsilongreedy import EpsilonGreedy
from ucb import UCB
from ally_elo import ELO
from JudgingSimulator import JudgingSimulator
import matplotlib.pyplot as plt

true_q = np.linspace(0.05, 0.95, 5)
n_judges = 1
opinion_var = 0.001
eval_var = 0.001
sim = JudgingSimulator(true_q, n_judges, opinion_var, eval_var) 

ucb = UCB(sim, c=1)
egreedy = EpsilonGreedy(sim, epsilon=0.1)
elo = ELO(sim, K=40)
ucb.run_experiment(n_episodes=20, top_n=-1, n_comparisons=50).print_results()#.plot('inclusion')
egreedy.run_experiment(n_episodes=20, top_n=-1,n_comparisons=50).print_results()#.plot('inclusion')
elo.run_experiment(n_episodes=20, top_n=-1, n_comparisons=50).print_results()#.plot('inclusion')
