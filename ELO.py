from AbstractAlgo import AbstractAlgo
from JudgingSimulator import JudgingSimulator
import numpy as np
import random

# https://www.geeksforgeeks.org/elo-rating-algorithm/
# Python 3 program for Elo Rating
import math
  
# Function to calculate the Probability
def Probability(rating1, rating2):
    return 1.0 * 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (rating1 - rating2) / 400))
  
  
# Function to calculate Elo rating
# K is a constant.
# d determines whether
# Player A wins or Player B. 
def EloRating(Ra, Rb, K, d):
  
    # To calculate the Winning
    # Probability of Player B
    Pb = Probability(Ra, Rb)
  
    # To calculate the Winning
    # Probability of Player A
    Pa = Probability(Rb, Ra)
  
    # Case -1 When Player A wins
    # Updating the Elo Ratings
    if (d == 1) :
        Ra = Ra + K * (1 - Pa)
        Rb = Rb + K * (0 - Pb) 
  
    # Case -2 When Player B wins
    # Updating the Elo Ratings
    else :
        Ra = Ra + K * (0 - Pa)
        Rb = Rb + K * (1 - Pb)

    return Ra, Rb
  
    # print("Updated Ratings:-")
    # print("Ra =", round(Ra, 6)," Rb =", round(Rb, 6))

class ELO(AbstractAlgo):
    def __init__(self):
        super().__init__()

        self.K = 30
        self.d = 1

    def rank_teams(self, n_teams, n_judges, true_q, c, var, top_n):
        prev_teams  = [i for i in range(n_judges)]
        team_scores = [1000 for _ in range(n_teams)]

        sim = JudgingSimulator(true_q, n_judges, var, var)

        steps = 10

        t = 0
        while t < steps:
            for j in range(n_judges):                
                t1 = prev_teams[j]

                t2 = None
                closest_dif = 9999 # simulated inf
                for team in range(n_teams):
                    if t1 != team and abs(team_scores[t1] - team_scores[team]) < closest_dif:
                        closest_dif = abs(team_scores[t1] - team_scores[team])
                        t2 = team        

                winner = sim.judge(j, t1, t2)
                loser = t2 if winner == t1 else t1

                team_scores[winner], team_scores[loser] = EloRating(team_scores[winner], team_scores[loser], self.K, self.d)
                
                prev_teams[j] = t2

            t += 1

        return np.argsort(team_scores)[::-1], t, None

    def generate_plots(self):
        return super().generate_plots()

    def __str__(self):
        return "***************** ELO Algorithm *****************"