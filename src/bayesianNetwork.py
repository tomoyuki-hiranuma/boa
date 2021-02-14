import pandas as pd
import numpy as np
from pgmpy.models import BayesianModel
from copy import deepcopy
from pgmpy.estimators import K2Score, BicScore, HillClimbSearch, ExhaustiveSearch
from pgmpy.sampling import BayesianModelSampling
from src.population import Population

class BayesianNetwork:
  def __init__(self, individual_array):
    self.network = None
    self.model = None
    self.data = self._to_DataFrame(individual_array)
    self.nodes = None

  def construct_network_by_k2_algorithm(self):
    BIC_tables = self.create_bic_tables()
    network = []
    '''
      tableの上から順にノードとして追加
      貪欲法でスコア順に追加しつつ、すべての遺伝子組のスコアが負になるまでネットワーク構
    '''
    return network

  def estimate(self):
    self.network = self.construct_network_by_k2_algorithm()
    # estimated_network = HillClimbSearch(self.data)
    # self.network = estimated_network.estimate(max_indegree=2)

  def fit(self):
    self.estimate()
    self.model = BayesianModel(list(self.network.edges()))
    self.model.add_nodes_from(list(self.network.nodes()))
    self.model.fit(self.data)

  def create_bic_tables(self):
    ## 前処理で(ペア, スコア)のテーブルを作る
    ## self.dataに対して、全ての組み合わせのBICを計算
    tables = []
    bic = BicScore(self.data)
    for par_label, par_items in self.data.iteritems():
      for child_label, child_items in self.data.iteritems():
        if par_label != child_label:
          model = BayesianModel([(par_label, child_label)])
          print("parent: {}, child: {}".format(par_label, child_label))
          print("BIC: {}".format(bic.score(model)))
          tables.append((par_label, child_label, bic.score(model)))
          print("-----------\n")
    sorted_tables = sorted(tables, key=lambda x: x[2])[::-1]
    return sorted_tables

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
    # リファクタリング余地あり
    array = []
    for individual in individual_array:
      array.append(list(individual.gene))
    return np.array(array)

  def _to_DataFrame(self, data):
    data = self._to_numpy_array(data)
    pd_data = pd.DataFrame()
    node_columns = []
    for index in range(len(data[0])):
      column_name = "X" + str(index+1)
      pd_data[column_name] = data.T[index]
      node_columns.append(column_name)
    self.nodes = node_columns
    return pd_data

if __name__ == '__main__':
  N = 4
  POP_SIZE = 5
  pop1 = Population(POP_SIZE, N)
  # print(pop1)
  BN = BayesianNetwork(pop1.array)
  bic_tables = BN.create_bic_tables()
  print(np.array(bic_tables))
  # print(BN.data)
  # BN.estimate()
  # BN.fit()
  # cpds = BN.model.get_cpds()
  # for cpd in cpds:
  #   print(cpd)
  # new_data = BN.sample_data(new_data_size=20)
  # print(new_data)