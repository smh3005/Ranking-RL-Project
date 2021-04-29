from abc import ABC, abstractmethod
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
import rbo
import math

ordinal_names = ['first', 'second', 'third', 'fourth', 'fifth']


class AbstractAlgo(ABC):
    def __init__(self, sim):
        """
        _true_order: smallest q is last place (largest order). largest q is order=0.
        """
        self.sim = sim
        self._true_order = np.argsort(self.sim.true_q)[::-1]
        self._ncomparisons = []
        self._ranks = []

    @property
    def n_comparisons(self):
        return self._ncomparisons

    @property
    def n_episodes(self):
        assert len(self._ncomparisons) == len(self._ranks)
        return len(self._ranks)

    @property
    def accuracies(self):
        is_equal = (np.array(self._ranks) ==
                    np.tile(self._true_order, (self.n_episodes, 1)))
        return is_equal[:, self._true_order]

    @property
    def inclusions(self):
        c2 = Counter(tuple(x) for x in iter(self._ranks))
        for order in self._true_order:
            inclusion = sum(
                [c2[x] for x in c2 if order in x[0:5]]) / self.n_episodes
            yield inclusion

    def rbo(self, p):
        rbos = np.empty(self.n_episodes)
        for i, rank in enumerate(self._ranks):
            rbos[i] = rbo.RankingSimilarity(self._true_order, rank).rbo(p=p)
        return rbos.mean()

    @abstractmethod
    def rank_teams(self, top_n, n_comparisons):
        """
        top_n: will converge after the q values for the top_n are steady
        """
        pass

    @abstractmethod
    def get_plot_name(self):
        raise NotImplementedError

    def run_experiment(self, n_episodes, top_n=None, n_comparisons=None):
        """
        if top_n is not None, stop iteration when the order in the top_n positions has not changed
        if top_n is None, set top_n = 5
        if top_n is -1, this stopping criteria will be neglected
        if n_comparisons is not None, stop iteration when we have done n_comparisons
        if n_comparisons is None, set n_comparisons=nlog(n)
        Stop when either condition is satisfied
        """
        if n_comparisons is None:
            n_comparisons = self.sim.n_teams * math.log2(self.sim.n_teams)
        if top_n is None:
            top_n = 5

        ranks = []
        times = []
        for _ in range(n_episodes):
            rank, time = self.rank_teams(top_n, n_comparisons)
            ranks.append(rank)
            times.append(time)
        self._ranks.extend(ranks)
        self._ncomparisons.extend(times)
        return self

    def print_results(self, nplaces=5):
        ''' CONVERGENCE METRICS '''
        print(self)
        print("Average number of time steps until convergence: ")
        print(np.mean(self.n_comparisons))
        print()

        print('Accuracy: ')
        for name, accuracy in zip(ordinal_names[:nplaces], self.accuracies.mean(axis=0)):
            print(f"{name}: {int(round(accuracy*100))}%")

        print('Inclusion in Top-5: ')
        for name, inclusion in zip(ordinal_names[:nplaces], self.inclusions):
            print(f"{name}: {int(round(inclusion*100))}%")

        return self

    def plot(self, plottype='accuracy', nplaces=5):
        if plottype == 'accuracy':
            xs = np.arange(nplaces) + 1
            plt.xticks(xs)
            accuracies = self.accuracies[:, :nplaces]
            mean = accuracies.mean(axis=0)
            std = accuracies.std(axis=0)
            plt.plot(xs, mean, label=self.get_plot_name())
            plt.fill_between(xs, mean-std*0.4, mean+std*0.4, alpha=0.2)
            plt.xlabel('Place')
            plt.ylabel('Accuracy')
        elif plottype == 'inclusion':
            xs = np.arange(nplaces) + 1
            plt.xticks(xs)
            inclusions = [x for i, x in enumerate(
                self.inclusions) if i < nplaces]
            plt.plot(xs, inclusions, label=self.get_plot_name())
            plt.xlabel('Place')
            plt.ylabel('Inclusion')
        else:
            raise NotImplementedError

        return self

    @abstractmethod
    def __str__(self):
        pass
