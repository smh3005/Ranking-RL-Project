from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict, Counter

class AbstractAlgo(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def rank_teams(self, n_teams, n_judges, true_q, c, var, top_n):
        pass

    @abstractmethod
    def generate_plots(self):
        pass

    def run_experiment(self, n_teams, n_judges, true_q, c, var, n_episodes, top_n):
        print(self)
        ranks = []
        times = []
        for n in range(n_episodes):
            rank, time, visit = self.rank_teams(n_teams, n_judges, true_q, c, var, top_n)
            ranks.append(rank)
            times.append(time)
            
        print("Average number of time steps until convergence: ")
        print(np.mean(times))
        print()

        print("Accuracy of first place winner: ")
        c1 = Counter(tuple(x)[0] for x in iter(ranks))
        print(str(c1[tuple(np.argsort(true_q))[::-1][0]]/n_episodes * 100) + "%")
        print()

        if n_teams > 5:
            print('Percent of time included in the top 5: ')
            c2 = Counter(tuple(x) for x in iter(ranks))
            first = sum([c2[x] for x in c2 if np.argsort(true_q)[::-1][0] in x[0:5]]) / n_episodes
            print("first: " + str(first * 100) + "%")
            second = sum([c2[x] for x in c2 if np.argsort(true_q)[::-1][1] in x[0:5]]) / n_episodes
            print("second: " + str(second * 100) + "%")
            third = sum([c2[x] for x in c2 if np.argsort(true_q)[::-1][2] in x[0:5]]) / n_episodes
            print("third: " + str(third * 100)+ "%")
            fourth = sum([c2[x] for x in c2 if np.argsort(true_q)[::-1][3] in x[0:5]]) / n_episodes
            print("fourth: " + str(fourth * 100) + "%")
            fifth = sum([c2[x] for x in c2 if np.argsort(true_q)[::-1][4] in x[0:5]]) / n_episodes
            print("fifth: " + str(fifth * 100) + "%")
        
        self.generate_plots()

    @abstractmethod
    def __str__(self):
        pass
    