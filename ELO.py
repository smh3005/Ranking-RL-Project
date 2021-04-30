# https://www.geeksforgeeks.org/elo-rating-algorithm/
# Python 3 program for Elo Rating
import math
import numpy as np
from scipy.stats import rankdata
from AbstractAlgo import AbstractAlgo

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
    if d == 1:
        Ra = Ra + K * (1 - Pa)
        Rb = Rb + K * (0 - Pb)

    # Case -2 When Player B wins
    # Updating the Elo Ratings
    else:
        Ra = Ra + K * (0 - Pa)
        Rb = Rb + K * (1 - Pb)

    return Ra, Rb

    # print("Updated Ratings:-")
    # print("Ra =", round(Ra, 6)," Rb =", round(Rb, 6))


class ELO(AbstractAlgo):
    def __init__(self, sim, K=30, d=1):
        super().__init__(sim)
        self.K = K
        self.d = d

    def rank_teams(self, top_n, n_comparisons):
        prev_teams = [[i] for i in range(self.sim.n_judges)]
        team_scores = [1000 for _ in range(self.sim.n_teams)]

        last_ranking = ([i for i in range(self.sim.n_teams)])

        n = 0
        t = 0
        while True:
            for j in range(self.sim.n_judges):
                tp1 = prev_teams[j][0]
                tp2 = None
                if len(prev_teams[j]) > 1:
                    tp2 = prev_teams[j][1]

                t2 = None
                closest_dif = 9999  # simulated inf
                for team in range(self.sim.n_teams):
                    if team != tp1 and team != tp2 and abs(team_scores[tp1] - team_scores[team]) < closest_dif:
                        closest_dif = abs(team_scores[tp1] - team_scores[team])
                        t2 = team

                winner = self.sim.judge(j, tp1, t2)
                loser = t2 if winner == tp1 else tp1

                team_scores[winner], team_scores[loser] = EloRating(
                    team_scores[winner], team_scores[loser], self.K, self.d)

                if len(prev_teams[j]) == 1:
                    prev_teams[j].append(tp1)
                elif len(prev_teams[j]) == 2:
                    prev_teams[j][1] = tp1
                else:
                    raise Exception("prev team history is too long... something's wrong")
                prev_teams[j][0] = t2

                n += 1

            new_ranking = rankdata(team_scores)[::-1]

            done = (top_n > 0 and (last_ranking[:top_n] == new_ranking[:top_n]).all()) or \
                   (n_comparisons > 0 and t >= n_comparisons)

            if done:
                break

            last_ranking = [i for i in new_ranking]

            t += 1

        rank = rankdata(-np.array(team_scores), method='min')-1
        return rank, t

    def get_plot_name(self):
        return 'ELO'

    def __str__(self):
        return "***************** ELO Algorithm *****************"
