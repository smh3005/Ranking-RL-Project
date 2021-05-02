# https://www.geeksforgeeks.org/elo-rating-algorithm/
# Python 3 program for Elo Rating

import math
import random
from collections import defaultdict
from queue import Queue

import numpy as np
from scipy.stats import rankdata

from AbstractAlgo import AbstractAlgo

class ELO(AbstractAlgo):
    def __init__(self, sim, K=30, d=1):
        super().__init__(sim)
        self.K = K

    def random_tiebreaker_golf(self, scores):
        rank = rankdata(np.array(scores), method='min')-1
        tiebreaker = []
        for i in range(max(rank)+1):
            ties = np.where(rank == i)[0]
            np.random.shuffle(ties)
            for x in ties:
                tiebreaker.append(x)
        return tiebreaker

    def random_tiebreaker(self, scores):
        rank = rankdata(-np.array(scores), method='min')-1
        tiebreaker = []
        for i in range(max(rank)+1):
            ties = np.where(rank == i)[0]
            np.random.shuffle(ties)
            for x in ties:
                tiebreaker.append(x)
        return tiebreaker
    

    def elo(self, Q, prev2):
        diff = [abs(Q[x] - Q[prev2[1]]) for x in range(self.sim.n_teams)]
        tiebreaker = self.random_tiebreaker_golf(diff)
        curr = tiebreaker[0]
        i = 1
        while curr in prev2:
            curr = tiebreaker[i]
            i += 1
        return curr
    
    def probability(self, Q, prev, curr):
        p_prev = 1.0 * 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (Q[curr] - Q[prev]) / 400))
        p_curr = 1.0 * 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (Q[prev] - Q[curr]) / 400))
        return p_prev, p_curr

    def update(self, Q, N, curr, prev, winner, t, top_n, n_comparisons):
        N[prev] += 1
        N[curr] += 1

        p_prev, p_curr = self.probability(Q, prev, curr)

        if winner == curr: 
            R = 1
        else:
            R = 0

        last_ranking = rankdata(-np.array(Q), method='min')-1

        Q[prev] = Q[prev] + self.K * (int(not R) - p_prev)
        Q[curr] = Q[curr] + self.K * (int(R) - p_curr)

        new_ranking = rankdata(-np.array(Q), method='min')-1

        done = (top_n > 0 and (last_ranking[:top_n] == new_ranking[:top_n]).all()) or \
            (n_comparisons > 0 and t >= n_comparisons)

        return done
        
    def rank_teams(self, top_n, n_comparisons):
        Q = [1000 for _ in range(self.sim.n_teams)]
        N = [0 for _ in range(self.sim.n_teams)]
        t = 0

        judge_current = defaultdict(int)
        judge_previous = defaultdict(list)

        judge_queue = Queue()
        for j in range(self.sim.n_judges):
            judge_queue.put(j)
            judge_previous[j] = [None, random.randint(0, self.sim.n_teams-1)]

        while True:
            j = judge_queue.get()
            t += 1
            judge_current[j] = self.elo(Q, judge_previous[j])
            winner = self.sim.judge(j, judge_current[j], judge_previous[j][1])
            done = self.update(Q, N, judge_current[j], judge_previous[j][1], winner, t, top_n, n_comparisons)

            if done:  # return the sorted list if q-values have converged
                rank = rankdata(1-np.array(Q), method='min')-1
                return rank, t
                
            judge_previous[j][0] = judge_previous[j][1]
            judge_previous[j][1] = judge_current[j]
            judge_queue.put(j)  # assign curr to prev


    def get_plot_name(self):
        return 'ELO'

    def __str__(self):
        return "***************** ELO Algorithm *****************"
