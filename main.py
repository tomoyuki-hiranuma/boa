import numpy as np
import random
import pandas as pd
from pgmpy.models import BayesianModel
from copy import deepcopy
from pgmpy.estimators import K2Score, BicScore, HillClimbSearch, ExhaustiveSearch
from pgmpy.sampling import BayesianModelSampling

def select(population, _size):
  eval_pop = np.array([sum(x) for x in population])
  sorted_indexes = np.argsort(eval_pop)
  sorted_array = np.zeros([len(population), len(population[0])])
  for index1, index2 in enumerate(sorted_indexes):
    sorted_array[index1] = population[index2]
  sorted_array = sorted_array[::-1]
  return sorted_array[0:_size]

def calc_evaluation(self):
  self.population.calc_evaluation(fucntion)

def sort_pop(population):
  eval_pop = np.array([sum(x) for x in population])
  sorted_indexes = np.argsort(eval_pop)
  sorted_array = np.zeros([len(population), len(population[0])])
  for index1, index2 in enumerate(sorted_indexes):
    sorted_array[index1] = population[index2]
  sorted_array = sorted_array[::-1]
  return sorted_array

def generate_new_population(population, data):
  new_population = sort_pop(population)
  for index in range(len(data)):
    # print("swap: {} , {}".format(new_population[len(population) - 1 - index], data[index]))
    new_population[len(population) - 1 - index] = deepcopy(data[index])
  return new_population

if __name__ == '__main__':
  np.random.seed(0)
  random.seed(0)

  individual_size = 10
  pop_size = 100
  select_pop_size = 30
  prob_one = 0.5
  new_data_size = 30
  generation = 0

  population = np.random.choice([0, 1], [pop_size, individual_size], p=[1.0 - prob_one, prob_one])

  while generation < 10:
    selected_population = select(population, select_pop_size)
    # generate data set as DataFrame
    data = pd.DataFrame()
    for index in range(individual_size):
      column_name = "X" + str(index+1)
      data[column_name] = selected_population.T[index]

    # create Network from data by HillClimbSearch, BicScore
    network = HillClimbSearch(data, scoring_method=BicScore(data))
    best_model = network.estimate() # DAGクラス
    edges = list(best_model.edges())
    nodes = list(best_model.nodes())
    # print(edges)
    # print(nodes)

    # create network as BayesianModel class
    bayesian_model = BayesianModel(edges)
    bayesian_model.add_nodes_from(nodes)
    bayesian_model.fit(data)
    # cpds = bayesian_model.get_cpds()
    # for cpd in cpds:
      # print(cpd, "\n")

    # create Model to sample data
    inference = BayesianModelSampling(bayesian_model)
    # sort by columns
    new_data = inference.forward_sample(size=new_data_size, return_type='dataframe')
    list_col_sorted = new_data.columns.to_list()
    list_col_sorted.sort()
    new_data = new_data.loc[:, list_col_sorted]
    new_data = new_data.to_numpy()

    # generate new population
    new_population = generate_new_population(population, new_data)
    # print(new_population)
    population = deepcopy(new_population)
    generation += 1

  print(population.mean())