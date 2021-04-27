from AbstractAlgo import AbstractAlgo
from JudgingSimulator import JudgingSimulator
import numpy as np
import trueskill
import random

class ELO(AbstractAlgo):
    def __init__(self):
        super().__init__()

    def rank_teams(self, n_teams, n_judges, true_q, c, var):
        prev_teams  = [-1 for _ in range(n_judges)]
        team_scores = [trueskill.Rating(2) for _ in range(n_teams)]

        sim = JudgingSimulator(true_q, n_judges, var, var)

        num_comparisons = 0

        free_teams = set(range(n_teams))
        steps = 50

        t = 0
        while t < steps:
            for j in range(n_judges):
                
                t1 = prev_teams[j]
                if t1 == -1:
                    prev_teams[j] = free_teams.pop()
                    continue

                t2 = None
                closest_dif = 9999 # simulated inf
                for team in free_teams:
                    if t2 == None:
                        t2 = team
                    elif abs(trueskill.quality_1vs1(team_scores[t2], team_scores[team]) - 0.5) < closest_dif:
                        closest_dif = abs(trueskill.quality_1vs1(team_scores[t2], team_scores[team]) - 0.5)
                        t2 = team        

                winner = sim.judge(j, t1, t2)
                loser = t2 if winner == t1 else t1

                team_scores[t1], team_scores[t2] = trueskill.rate_1vs1(team_scores[winner], 
                                                                       team_scores[loser], min_delta=0.1)
                
                free_teams.add(t1)
                prev_teams[j] = t2
                free_teams.remove(t2)

            t += 1

        print(team_scores)
        print(np.argsort(team_scores[::-1]))
        print(true_q)
        exit()
        return np.argsort(team_scores[::-1]), t, None

    def generate_plots(self):
        return super().generate_plots()

    def __str__(self):
        return "***************** ELO Algorithm *****************"