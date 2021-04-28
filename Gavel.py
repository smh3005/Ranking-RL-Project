import math
from random import random

import numpy as np
from numpy.random import shuffle
from recordtype import recordtype

import crowd_bt
from AbstractAlgo import AbstractAlgo
from JudgingSimulator import JudgingSimulator

Team = recordtype("Team", "mu sigma_sq id")
Judge = recordtype("Judge", "alpha beta prev next")


def choose_next(judge, teams):
    shuffle(teams)  # useful for argmax case as well in the case of ties
    if teams:
        if random() < crowd_bt.EPSILON:
            return teams[0]
        else:
            return crowd_bt.argmax(lambda i: crowd_bt.expected_information_gain(
                judge.alpha,
                judge.beta,
                judge.prev.mu,
                judge.prev.sigma_sq,
                i.mu,
                i.sigma_sq), teams)
    else:
        return None


def perform_vote(judge, next_won):
    if next_won:
        winner = judge.next
        loser = judge.prev
    else:
        winner = judge.prev
        loser = judge.next
    u_alpha, u_beta, u_winner_mu, u_winner_sigma_sq, u_loser_mu, u_loser_sigma_sq = crowd_bt.update(
        judge.alpha,
        judge.beta,
        winner.mu,
        winner.sigma_sq,
        loser.mu,
        loser.sigma_sq
    )
    judge.alpha = u_alpha
    judge.beta = u_beta
    winner.mu = u_winner_mu
    winner.sigma_sq = u_winner_sigma_sq
    loser.mu = u_loser_mu
    loser.sigma_sq = u_loser_sigma_sq


class Gavel(AbstractAlgo):
    def rank_teams(self, n_teams, n_judges, true_q, c, var, top_n):
        start_mu = 20
        start_sig = 1
        teams = [Team(start_mu, start_sig, i) for i in range(n_teams)]
        judges = [Judge(var, var, teams[i], None) for i in range(n_judges)]
        sim = JudgingSimulator(true_q, n_judges, var, var)

        last_ranking = ([i for i in range(n_teams)])[:top_n]

        n = 0
        t = 0
        while True:
            team_scores = [t.mu for t in teams]
            for j_id, j in enumerate(judges):
                j.next = choose_next(j, [t for t in teams if t != j.prev])

                winner = sim.judge(j_id, j.prev.id, j.next.id)
                perform_vote(j, teams[winner] == j.next)

                j.prev = j.next
                team_scores[j.prev.id] = j.prev.mu
                team_scores[j.next.id] = j.next.mu
                n += 1

            new_ranking = (np.argsort(team_scores)[::-1])[:top_n]
            if (new_ranking == last_ranking).all() and t > n_teams * math.log2(n_teams):
                break
            last_ranking = [i for i in new_ranking]

            t += 1

        return new_ranking, t

    def __str__(self):
        return "***************** Gavel Algorithm *****************"
