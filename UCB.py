from AbstractAlgo import AbstractAlgo
from JudgingSimulator import JudgingSimulator
import numpy as np

class UCB(AbstractAlgo):

    def __init__(self):
        super(UCB).__init__()
        
    ''' UCB  
    input: q-value per team (Q), visits per team (N), number of teams (n_teams),
    previous team (prev), exploration constant (c), time (t)
    output: a team to visit (curr)`
    '''
    def ucb(self, Q, N, n_teams, prev, c, t):
        # calculate UCB score for each team
        ucb_scores = []
        for a in range(n_teams):
            if N[a] == 0:
                ucb_scores.append(float('inf'))
            else:
                ucb_scores.append(Q[a] + c * np.sqrt(np.log(2*t)/N[a]))
        # choose the team with the highest UCB score
        curr = np.argmax(ucb_scores)
        # make sure that curr != prev
        if curr == prev:
            curr = np.argsort(ucb_scores)[-2]
        
        return curr

    ''' UPDATE
    input: q-value per team (Q), visits per team (N), current team (curr), 
    previous team (prev), winner (winner), time (t)
    output: boolean indicating whether the q-values have converged
    '''
    def update(self, Q, N, curr, prev, winner, t):
        N[curr] += 1 # visit current team
        N[prev] += 1 # visit previous team
        if winner == curr: # assign R=1 if curr is the winner, else R=0
            R = 1
        else:
            R = 0
        curr_Q = Q[curr] # get current team's q-value
        prev_Q = Q[prev] # get previous team's q-value

        Q[curr] = curr_Q + 1/N[curr] * (int(R) - curr_Q) # incremental average to update current team's q-value
        Q[prev] = prev_Q + 1/N[prev] * (int(not R) - prev_Q) # incremental average to update previous team's q-value
        if abs(Q[curr] - curr_Q)  < 0.01 and abs(Q[prev] - prev_Q) < 0.01 and min(N) > 0: # check for convergence
            return True
        else:
            return False

    def rank_teams(self, n_teams, n_judges, true_q, c, var):
        Q = [0 for _ in range(n_teams)] # intialize q-values (e.g. q(·)=0 for all teams)
        N = [0 for _ in range(n_teams)] # initialize number of visits (e.g. n(·)=0 for all teams)
        t = 0 # initialize time (e.g. t=0)
        sim = JudgingSimulator(true_q, n_judges, var, var) # initialize simulator
        prev = self.ucb(Q,N,n_teams,None,c,t) # dispatch a judge to visit the first team
        while True: # WHILE q-values haven't converged
            t += 1 # increment time
            curr = self.ucb(Q,N,n_teams,prev,c,t) # dispatch a judge to visit a team
            winner = sim.judge(0,curr,prev) # simulate the judge's decision
            # update the team's scores
            done = self.update(Q,N,curr,prev,winner,t) # update the team's q-values
            if done: # return the sorted list if q-values have converged
                return np.argsort(Q)[::-1], t, N
            prev = curr # assign curr to prev      

    def generate_plots(self):
        return super().generate_plots()  

    def __str__(self):
        return "***************** UCB Algorithm *****************"