from AbstractAlgo import AbstractAlgo
from JudgingSimulator import JudgingSimulator
from scipy.stats import rankdata
import math
import numpy as np
import random
from collections import defaultdict
from collections import Counter
from queue import Queue
import rbo
from copy import deepcopy
import matplotlib.pyplot as plt

class EGREEDY_PREV2(AbstractAlgo):

    def __init__(self):
        super(EGREEDY_PREV2).__init__()
        
    ''' EGREEDY  
    input: q-value per team (Q), visits per team (N), number of teams (n_teams),
    judge's previous 2 teams (prev2), exploration constant (c), time (t)
    output: a team to visit (curr)`
    '''
    def egreedy(self, Q, n_teams, prev2, c, t):
        if random.random() > c: # choose highest scoring team with probability 1-epsilon
            curr = np.random.choice(np.flatnonzero(np.array(Q) == np.array(Q).max()))
            i = 2
            while curr in prev2:
                curr = np.argsort(Q)[-i]
                i += 1
        else: # choose random team with proability epsilon
            curr = random.randint(0,n_teams-1)
            while curr in prev2: # make sure that cur != prev
                curr = random.randint(0,n_teams-1)
        return curr

    ''' UPDATE
    input: q-value per team (Q), visits per team (N), judge's current team (curr), 
    judge's previous teams (prev), winner (winner), time (t), top n teams (top_n)
    output: boolean indicating whether the top_n ranks have converged
    '''
    def update(self, Q, N, curr, prev, winner, t, top_n):
        N[curr] += 1 # visit current team
        N[prev] += 1 # visit previous team
        if winner == curr: # assign R=1 if curr is the winner, else R=0
            R = 1
        else:
            R = 0

        curr_Q = Q[curr] # get current team's q-value
        prev_Q = Q[prev] # get previous team's q-value

        if N[curr] == 1:
            curr_Q = 0
        if N[prev] == 1:
            prev_Q = 0

        last_ranking = rankdata(Q[::-1], method='min')[:top_n]

        Q[curr] = curr_Q + 1/N[curr] * (int(R) - curr_Q) # incremental average to update current team's q-value
        Q[prev] = prev_Q + 1/N[prev] * (int(not R) - prev_Q) # incremental average to update previous team's q-value

        new_ranking = rankdata(Q[::-1], method='min')[:top_n]

        if (last_ranking == new_ranking).all() and t > len(N) * math.log2(len(N)):
            return True
        else:
            return False

    def rank_teams(self, n_teams, n_judges, true_q, c, var, top_n):
        Q = [1 for _ in range(n_teams)] # intialize q-values (e.g. q(·)=0 for all teams)
        N = [0 for _ in range(n_teams)] # initialize number of visits (e.g. n(·)=0 for all teams)
        t = 0 # initialize time (e.g. t=0)
        sim = JudgingSimulator(true_q, n_judges, var, var) # initialize simulator

        judge_current = defaultdict(int)
        judge_previous = defaultdict(list)

        judge_queue = Queue()
        for j in range(n_judges):
            judge_queue.put(j)
            judge_previous[j] = [None, random.randint(0, n_teams-1)]

        while True: # WHILE q-values haven't converged
            j = judge_queue.get()
            t += 1 # increment time
            judge_current[j] = self.egreedy(Q, n_teams, judge_previous[j], c, t) # dispatch a judge to visit a team
            winner = sim.judge(0,judge_current[j],judge_previous[j][1]) # simulate the judge's decision
            done = self.update(Q,N,judge_current[j],judge_previous[j][1],winner,t,top_n) # update the team's q-values
            if done: # return the sorted list if q-values have converged
                return np.argsort(Q)[::-1], t
            judge_previous[j][0] = judge_previous[j][1]
            judge_previous[j][1] = judge_current[j]
            judge_queue.put(j) # assign curr to prev      

    def generate_plots(self):
        return super().generate_plots()  

    def __str__(self):
        return "***************** EGREEDY PREV 2 Algorithm *****************"