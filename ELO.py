from AbstractAlgo import AbstractAlgo
from JudgingSimulator import JudgingSimulator
import numpy as np
import trueskill

# Simulation Parameters
# opinion_var = 0.01
# eval_var    = 0.01
# num_teams   = 10
# n_judges  = 1
# teams_var   = 1

class ELO(AbstractAlgo):
    def __init__(self):
        super().__init__()

    def rank_teams(self, n_teams, n_judges, true_q, c, var):
        prev_teams  = [-1 for _ in range(n_judges)]
        team_scores = [trueskill.Rating(1000) for _ in range(n_teams)]

        sim = JudgingSimulator(true_q, n_judges, var, var)

        num_comparisons = 0

        free_teams = set(range(n_teams))
        steps = 100

        t = 0
        while t < steps:
            for j in range(n_judges):
                t1 = prev_teams[j]
                t2 = free_teams.pop() # TODO: make sure random

                if t1 == -1:
                    prev_teams[j] = t2
                    continue

                winner = sim.judge(j, t1, t2)
                loser = t2 if winner == t1 else t1

                team_scores[t1], team_scores[t2] = trueskill.rate_1vs1(team_scores[winner], 
                                                                       team_scores[loser])
                
                free_teams.add(t1)
                prev_teams[j] = t2

            t += 1

        return np.argsort(team_scores[::-1]), t, None

    def generate_plots(self):
        return super().generate_plots()

    def __str__(self):
        return "***************** ELO Algorithm *****************"