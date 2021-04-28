from abc import ABC, abstractmethod
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


class AbstractAlgo(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def rank_teams(self, n_teams, n_judges, true_q, c, var, top_n):
        pass

    @abstractmethod
    def get_plot_name(self):
        raise NotImplementedError

    def generate_plots(self, kwargs):
        pass

    def run_experiment(self, n_teams, n_judges, true_q, c, var, n_episodes, top_n):
        print(self)
        ranks = []
        times = []
        for _ in range(n_episodes):
            rank, time = self.rank_teams(
                n_teams, n_judges, true_q, c, var, top_n)
            ranks.append(rank)
            times.append(time)

        ''' CONVERGENCE METRICS '''
        print("Average number of time steps until convergence: ")
        print(np.mean(times))
        print()

        ''' ACCURACY METRICS '''
        print('Accuracy: ')
        c1 = Counter(tuple(x)[0] for x in iter(ranks))
        first = c1[tuple(np.argsort(true_q))[::-1][0]]/n_episodes
        print("first: " + str(first*100) + "%")

        c1 = Counter(tuple(x)[1] for x in iter(ranks))
        second = c1[tuple(np.argsort(true_q))[::-1][1]]/n_episodes
        print("second: " + str(second*100) + "%")

        c1 = Counter(tuple(x)[2] for x in iter(ranks))
        third = c1[tuple(np.argsort(true_q))[::-1][2]]/n_episodes
        print("third: " + str(third*100) + "%")

        c1 = Counter(tuple(x)[3] for x in iter(ranks))
        fourth = c1[tuple(np.argsort(true_q))[::-1][3]]/n_episodes
        print("fourth: " + str(fourth*100) + "%")

        c1 = Counter(tuple(x)[4] for x in iter(ranks))
        fifth = c1[tuple(np.argsort(true_q))[::-1][4]]/n_episodes
        print("fifth: " + str(fifth*100) + "%")
        print()

        ''' INCLUSION METRICS '''
        print('Inclusion in Top-5: ')
        c2 = Counter(tuple(x) for x in iter(ranks))
        first = sum([c2[x] for x in c2 if np.argsort(
            true_q)[::-1][0] in x[0:5]]) / n_episodes
        print("first: " + str(first * 100) + "%")
        second = sum([c2[x] for x in c2 if np.argsort(
            true_q)[::-1][1] in x[0:5]]) / n_episodes
        print("second: " + str(second * 100) + "%")
        third = sum([c2[x] for x in c2 if np.argsort(
            true_q)[::-1][2] in x[0:5]]) / n_episodes
        print("third: " + str(third * 100) + "%")
        fourth = sum([c2[x] for x in c2 if np.argsort(
            true_q)[::-1][3] in x[0:5]]) / n_episodes
        print("fourth: " + str(fourth * 100) + "%")
        fifth = sum([c2[x] for x in c2 if np.argsort(
            true_q)[::-1][4] in x[0:5]]) / n_episodes
        print("fifth: " + str(fifth * 100) + "%")

        self.generate_plots(locals())

    @abstractmethod
    def __str__(self):
        pass
