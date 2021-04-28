import math
import random
from collections import defaultdict
from queue import Queue

import numpy as np
from scipy.stats import rankdata

from AbstractAlgo import AbstractAlgo


class EpsilonGreedy(AbstractAlgo):

    ''' EGREEDY
    input: q-value per team (Q), visits per team (N), number of teams (n_teams),
    judge's previous 3 teams (prev3), exploration constant (c), time (t)
    output: a team to visit (curr)`
    '''
    def __init__(self, sim, epsilon):
        self._epsilon = epsilon
        super().__init__(sim)

    @property
    def epsilon(self):
        return self._epsilon

    def egreedy(self, Q, n_teams, prev3):
        if random.random() > self.epsilon:  # choose highest scoring team with probability 1-epsilon
            curr = np.random.choice(np.flatnonzero(
                np.array(Q) == np.array(Q).max()))
            i = 1
            while curr in prev3:
                curr = np.argsort(Q)[-i]
                i += 1
        else:  # choose random team with proability epsilon
            curr = random.randint(0, n_teams-1)
            while curr in prev3:  # make sure that cur != prev
                curr = random.randint(0, n_teams-1)
        return curr

    ''' UPDATE
    input: q-value per team (Q), visits per team (N), judge's current team (curr), 
    judge's previous teams (prev), winner (winner), time (t), top n teams (top_n)
    output: boolean indicating whether the top_n ranks have converged
    '''

    def update(self, Q, N, curr, prev, winner, t, top_n):
        N[curr] += 1  # visit current team
        N[prev] += 1  # visit previous team
        if winner == curr:  # assign R=1 if curr is the winner, else R=0
            R = 1
        else:
            R = 0

        curr_Q = Q[curr]  # get current team's q-value
        prev_Q = Q[prev]  # get previous team's q-value

        if N[curr] == 1:
            curr_Q = 0
        if N[prev] == 1:
            prev_Q = 0

        last_ranking = rankdata(Q[::-1], method='min')[:top_n]

        # incremental average to update current team's q-value
        Q[curr] = curr_Q + 1/N[curr] * (int(R) - curr_Q)
        # incremental average to update previous team's q-value
        Q[prev] = prev_Q + 1/N[prev] * (int(not R) - prev_Q)

        new_ranking = rankdata(Q[::-1], method='min')[:top_n]

        if (last_ranking == new_ranking).all() and t > len(N) * math.log2(len(N)):
            return True
        else:
            return False

    def rank_teams(self, top_n):
        """
        np.array([.5, .3, .1]).argsort()[::-1] -> array([0, 1, 2])
        """
        # intialize q-values (e.g. q(·)=0 for all teams)
        Q = [1 for _ in range(self.sim.n_teams)]
        # initialize number of visits (e.g. n(·)=0 for all teams)
        N = [0 for _ in range(self.sim.n_teams)]
        t = 0  # initialize time (e.g. t=0)

        judge_current = defaultdict(int)
        judge_previous = defaultdict(list)

        judge_queue = Queue()
        for j in range(self.sim.n_judges):
            judge_queue.put(j)
            judge_previous[j] = [None, None, random.randint(0, self.sim.n_teams-1)]

        while True:  # WHILE q-values haven't converged
            j = judge_queue.get()
            t += 1  # increment time
            # dispatch a judge to visit a team
            judge_current[j] = self.egreedy(
                Q, self.sim.n_teams, judge_previous[j])
            # simulate the judge's decision
            winner = self.sim.judge(0, judge_current[j], judge_previous[j][2])
            # update the team's q-values
            done = self.update(
                Q, N, judge_current[j], judge_previous[j][2], winner, t, top_n)
            if done:  # return the sorted list if q-values have converged
                return np.argsort(Q)[::-1], t
            judge_previous[j][0] = judge_previous[j][1]
            judge_previous[j][1] = judge_previous[j][2]
            judge_previous[j][2] = judge_current[j]
            judge_queue.put(j)  # assign curr to prev

    def get_plot_name(self):
        return f'$\\varepsilon$-greedy ($\\varepsilon={self.epsilon}$)'

    def __str__(self):
        return "***************** EGREEDY PREV 3 Algorithm *****************"
