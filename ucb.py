import math
import random
from collections import defaultdict
from queue import Queue

import numpy as np
from scipy.stats import rankdata

from AbstractAlgo import AbstractAlgo


class UCB(AbstractAlgo):
    ''' UCB
    input: q-value per team (Q), visits per team (N), number of teams (n_teams),
    judge's previous 2 teams (prev2), exploration constant (c), time (t)
    output: a team to visit (curr)`
    '''

    def __init__(self, sim, c):
        self.c = c
        super().__init__(sim)
    
    def random_tiebreaker(self, ucb_scores):
        rank = rankdata(-np.array(ucb_scores), method='min')-1
        tiebreaker = []
        for i in range(max(rank)+1):
            ties = np.where(rank == i)[0]
            np.random.shuffle(ties)
            for x in ties:
                tiebreaker.append(x)
        return tiebreaker

    def ucb(self, Q, N, n_teams, prev2, t):
        # calculate UCB score for each team
        ucb_scores = []
        for a in range(n_teams):
            if N[a] == 0:
                ucb_scores.append(float('inf'))
            else:
                ucb_scores.append(Q[a] + self.c * np.sqrt(np.log(2*t)/N[a]))
        # choose the team with the highest UCB score
        tiebreaker = self.random_tiebreaker(ucb_scores)
        curr = tiebreaker[0]
        # make sure that curr != prev
        i = 1
        while curr in prev2:
            curr = tiebreaker[i]
            i += 1
        return curr

    ''' UPDATE
    input: q-value per team (Q), visits per team (N), judge's current team (curr), 
    judge's previous teams (prev), winner (winner), time (t), top n teams (top_n)
    output: boolean indicating whether the top_n ranks have converged
    '''

    def update(self, Q, N, curr, prev, winner, t, top_n, n_comparisons):
        N[curr] += 1  # visit current team
        N[prev] += 1  # visit previous team
        if winner == curr:  # assign R=1 if curr is the winner, else R=0
            R = 1
        else:
            R = 0
        curr_Q = Q[curr]  # get current team's q-value
        prev_Q = Q[prev]  # get previous team's q-value

        last_ranking = rankdata(-np.array(Q), method='min')-1

        # incremental average to update current team's q-value
        Q[curr] = curr_Q + 1/N[curr] * (int(R) - curr_Q)
        # incremental average to update previous team's q-value
        Q[prev] = prev_Q + 1/N[prev] * (int(not R) - prev_Q)

        new_ranking = rankdata(-np.array(Q), method='min')-1

        done = (top_n > 0 and (last_ranking[:top_n] == new_ranking[:top_n]).all()) or \
            (n_comparisons > 0 and t >= n_comparisons)

        return done

    def rank_teams(self, top_n, n_comparisons):
        # intialize q-values (e.g. q(·)=0 for all teams)
        Q = [0 for _ in range(self.sim.n_teams)]
        # initialize number of visits (e.g. n(·)=0 for all teams)
        N = [0 for _ in range(self.sim.n_teams)]
        t = 0  # initialize time (e.g. t=0)

        judge_current = defaultdict(int)
        judge_previous = defaultdict(list)

        judge_queue = Queue()
        for j in range(self.sim.n_judges):
            judge_queue.put(j)
            judge_previous[j] = [None, random.randint(0, self.sim.n_teams-1)]

        while True:  # WHILE q-values haven't converged
            j = judge_queue.get()
            t += 1  # increment time
            # dispatch a judge to visit a team
            judge_current[j] = self.ucb(
                Q, N, self.sim.n_teams, judge_previous[j], t)
            # simulate the judge's decision
            winner = self.sim.judge(j, judge_current[j], judge_previous[j][1])
            # update the team's q-values
            done = self.update(
                Q, N, judge_current[j], judge_previous[j][1], winner, t, top_n, n_comparisons)

            if done:  # return the sorted list if q-values have converged
                rank = rankdata(1-np.array(Q), method='min')-1
                return rank, t
            judge_previous[j][0] = judge_previous[j][1]
            judge_previous[j][1] = judge_current[j]
            judge_queue.put(j)  # assign curr to prev

    def get_plot_name(self):
        return f'UCB (c={self.c})'

    def __str__(self):
        return "***************** UCB PREV-2 Algorithm *****************"
