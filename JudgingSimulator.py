from numpy.random import beta as Beta
import numpy as np


class JudgingSimulator:

    def __init__(self, expected_team_values, n_judges, opinion_var=0.02, eval_var=0.02):
        self.n_judges = n_judges
        self.eval_var = eval_var
        self.n_teams = len(expected_team_values)
        self.team_value_shapes = [self.get_shape_params(val, opinion_var)
                                  for val in expected_team_values]
        self.judges_opinions = [self._get_judges_opinions()
                                for _ in range(self.n_judges)]

    def _get_judges_opinions(self):
        # sample a judge's opinion of each team
        return [Beta(*value_shape) for value_shape in self.team_value_shapes]

    def judge(self, judge, team1, team2):
        # judge is the index of the judge we are querying
        # team1, team2 are the index of the two teams to compare
        # return the index of the team that wins
        eval1, eval2 = self.sample_evaluations(judge, [team1, team2])
        return team1 if eval1 > eval2 else team2

    def sample_evaluations(self, judge, teams=None):
        teams = teams or np.arange(self.n_teams)
        opinions = [self.judges_opinions[judge][team] for team in teams]
        eval_shapes = [self.get_shape_params(
            opinion, self.eval_var) for opinion in opinions]
        evals = [Beta(*eval_shape) for eval_shape in eval_shapes]
        return evals

    def get_variance(self, a, b):
        return (a*b) / ((a+b)**2 * (a+b+1))

    def get_shape_params(self, mean, var):
        a = -(mean*(var+mean**2-mean)) / var
        b = ((var + mean**2-mean)*(mean-1)) / var
        # make sure that a and b are positive
        return (max(0.0000001, a), max(0.0000001, b))

    def get_mean(self, a, b):
        return a / (a+b)
