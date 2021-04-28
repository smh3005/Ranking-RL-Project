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
        xs = np.arange(kwargs['nplaces']) + 1
        plt.xticks(xs)
        plt.plot(xs, kwargs['inclusions'], label=self.get_plot_name())
        plt.xlabel('Place')
        plt.ylabel('Inclusion')

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

        place_names = ['first', 'second', 'third', 'fourth', 'fifth']
        nplaces = len(place_names)

        accuracies = np.empty(nplaces)
        for i in range(nplaces):
            c = Counter(tuple(x)[i] for x in iter(ranks))
            accuracy = c[tuple(np.argsort(true_q))[::-1][i]] / n_episodes
            accuracies[i] = accuracy


        inclusions = np.empty(nplaces)
        c2 = Counter(tuple(x) for x in iter(ranks))
        for i in range(nplaces):
            inclusion = sum([c2[x] for x in c2 if np.argsort(
                true_q)[::-1][i] in x[0:5]]) / n_episodes
            inclusions[i] = inclusion

        print('Accuracy: ')
        for place, accuracy in zip(place_names, accuracies):
            print(f"{place}: {int(round(accuracy*100))}%")

        print('Inclusion in Top-5: ')
        for place, inclusion in zip(place_names, inclusions):
            print(f"{place}: {int(round(inclusion*100))}%")

        self.generate_plots(locals())

    @abstractmethod
    def __str__(self):
        pass
