import numpy as np
import random
import pandas as pd
from pgmpy.models import BayesianModel
from copy import deepcopy
from pgmpy.estimators import K2Score, BicScore, HillClimbSearch, ExhaustiveSearch

def select(population, _size):
  eval_pop = np.array([sum(x) for x in population])
  sorted_indexes = np.argsort(eval_pop)
  sorted_array = np.zeros([len(population), len(population[0])])
  for index1, index2 in enumerate(sorted_indexes):
    sorted_array[index1] = population[index2]
  sorted_array = sorted_array[::-1]
  return sorted_array[0:_size]

if __name__ == '__main__':
  np.random.seed(0)
  random.seed(0)

  individual_size = 4
  pop_size = 100
  select_pop_size = 50
  prob_one = 0.5

  population = np.random.choice([0, 1], [pop_size, individual_size], p=[1.0 - prob_one, prob_one])
  selected_population = select(population, select_pop_size)

  # generate data set as DataFrame
  data = pd.DataFrame()
  for index in range(individual_size):
    column_name = "X" + str(index+1)
    data[column_name] = selected_population.T[index]

  # create Network from data by HillClimbSearch, BicScore
  network = ExhaustiveSearch(data, scoring_method=BicScore(data))
  best_model = network.estimate() # DAGクラス
  edges = best_model.edges()
  nodes = best_model.nodes()
  print(edges)
  print(nodes)

  # # create network as BayesianModel class
  # bayesian_model = BayesianModel(best_model)
  # print(bayesian_model.nodes())
  