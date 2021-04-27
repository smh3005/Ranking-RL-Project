''' SIMULATOR '''
def get_shape_params(mean, var):
  a = -(mean*(var+mean**2-mean)) / var
  b = ((var + mean**2-mean)*(mean-1)) / var
  return (max(0.0000001,a), max(0.0000001,b)) ## make sure that a and b are positive

def get_variance(a, b):
  return (a*b) / ((a+b)**2 * (a+b+1))

def get_mean(a, b):
  return a / (a+b)

class JudgingSimulator:
  def __init__(self, expected_team_values, n_judges, opinion_var=0.02, eval_var=0.02):
    self.n_judges = n_judges
    self.eval_var = eval_var
    self.n_teams = len(expected_team_values)
    self.team_value_shapes = [get_shape_params(val, opinion_var)
                              for val in expected_team_values]

    def _get_judge_opinions():
      # sample a judge's opinion of each team
      return [Beta(*value_shape) for value_shape in self.team_value_shapes]

    self.judge_opinions = [_get_judge_opinions() for _ in range(n_judges)]

  def judge(self, judge, team1, team2):
    # judge is the index of the judge we are querying
    # team1, team2 are the index of the two teams to compare
    # return the index of the team that wins
    eval1, eval2 = self.sample_evaluations(judge, [team1, team2])
    return team1 if eval1 > eval2 else team2

  def sample_evaluations(self, judge, teams=None):
    teams = teams or np.arange(self.n_teams)
    opinions = [self.judge_opinions[judge][team] for team in teams]
    eval_shapes = [get_shape_params(opinion, self.eval_var) for opinion in opinions]
    evals = [Beta(*eval_shape) for eval_shape in eval_shapes]
    return evals