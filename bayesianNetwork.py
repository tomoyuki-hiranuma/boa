import pandas as pd
import numpy as np
from pgmpy.models import BayesianModel
from copy import deepcopy
from pgmpy.estimators import K2Score, BicScore, HillClimbSearch, ExhaustiveSearch
from pgmpy.sampling import BayesianModelSampling
from population import Population

class BayesianNetwork:
  def __init__(self, individual_array):
    self.network = None
    self.model = None
    self.data = self._to_DataFrame(individual_array)

  def estimate(self):
    estimated_network = HillClimbSearch(self.data, scoring_method=BicScore(self.data))
    self.network = estimated_network.estimate()

  def fit(self):
    self.model = BayesianModel(list(self.network.edges()))
    self.model.add_nodes_from(list(self.network.nodes()))
    self.model.fit(self.data)

  def sample_data(self, new_data_size):
    inference = BayesianModelSampling(self.model)
    sampled_data = inference.forward_sample(size=new_data_size, return_type='dataframe')
    sampled_data = self.get_data_sorted_by_columns(sampled_data)
    return sampled_data.to_numpy()

  def get_data_sorted_by_columns(self, data):
    list_col_sorted = data.columns.to_list()
    list_col_sorted.sort()
    return data.loc[:, list_col_sorted]

  def _to_numpy_array(self, individual_array):
    array = []
    for individual in individual_array:
      # print(individual.gene)
      array.append(individual.gene)
    return np.array(array)

  def _to_DataFrame(self, data):
    data = self._to_numpy_array(data)
    pd_data = pd.DataFrame()
    for index in range(len(data)-1):
      column_name = "X" + str(index+1)
      pd_data[column_name] = data.T[index]
    return pd_data

if __name__ == '__main__':
  N = 4
  POP_SIZE = 5
  pop1 = Population(POP_SIZE, N)
  # print(pop1)
  BN = BayesianNetwork(pop1.array)
  # print(BN.data)
  BN.estimate()
  BN.fit()
  cpds = BN.model.get_cpds()
  for cpd in cpds:
    print(cpd)
  new_data = BN.sample_data(new_data_size=20)
  print(new_data)