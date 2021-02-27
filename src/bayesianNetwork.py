import pandas as pd
import numpy as np
from pgmpy.models import BayesianModel
from copy import deepcopy
from math import log, log2
import networkx as nx
from pgmpy.estimators import K2Score, BicScore, HillClimbSearch, ExhaustiveSearch
from pgmpy.sampling import BayesianModelSampling
from population import Population

class BayesianNetwork:
  def __init__(self, individual_array, u):
    self.network = [] # 
    self.model = None # BayesianModelクラスのインスタ
    self.nodes = None # nodeのリスト
    self.data = self._to_DataFrame(individual_array) # pandas.DataFrame
    self.max_indegree = u
    self.metric = K2Score(self.data)

  def estimate_network_by_k2(self):
    self.network = self.construct_network_by_k2_algorithm()

  def estimate_network_by_hillclimb(self):
    estimater = HillClimbSearch(self.data)
    best_model = estimater.estimate(scoring_method=self.metric)
    self.network = list(best_model.edges())

  def fit(self):
    print("ネットワーク構築")
    self.estimate_network_by_k2()
    # 構築したネットワークのエッジを使う
    self.model = BayesianModel(self.network)
    # 独立なノードがあったとき
    self.model.add_nodes_from(self.nodes)
    self.model.fit(self.data)

    '''
      K2アルゴリズム
      【フロー】
      全てのノード間で最もBICスコアの高いノード間にエッジを張る（このときn個のノード、1つのエッジ）
      エッジを保存した状態で全てのノード間のBICスコアを計算し、最も高いノード間にエッジを張る(このときn個のノード、2つのエッジ)
      最終的にスコアが負になったら探索終了
    '''
  def construct_network_by_k2_algorithm(self):
    network = []
    for child_index in range(len(self.nodes)):
      child_node = self.nodes[child_index]
      # 空の親に対してスコア計算
      local_network = []
      old_model = BayesianModel()
      old_model.add_nodes_from(self.nodes)
      old_score = self.metric.score(old_model)
      ok_to_proceed = True
      while ok_to_proceed and len(local_network) < self.max_indegree:
        # 最大となる親ノード候補を抽出
        parent_candidate_node, parent_candidate_index = self.get_candidate_info(child_index)

        if parent_candidate_node == None:
          ok_to_proceed = False
          continue
        # print("candidate parent: ", parent_candidate_node, ", child:", child_node)
        new_model = BayesianModel(local_network)
        new_model.add_nodes_from(self.nodes)
        new_model.add_edge(parent_candidate_node, child_node)
        if self.metric.score(new_model) > self.metric.score(old_model):
          network.append([parent_candidate_node, child_node])
          local_network.append([parent_candidate_node, child_node])
          old_model = new_model.copy()
        else:
          ok_to_proceed = False
    print("selected network:", network)
    return network

  def get_candidate_info(self, child_index):
    child_node = self.nodes[child_index]
    max_score = -float('inf')
    selected_parent_node = None
    selected_parent_index = -1
    candidate_model = BayesianModel()
    candidate_model.add_nodes_from(self.nodes)
    for diff_index in range(len(self.nodes[child_index:])):
      parent_index = child_index + diff_index
      parent_node = self.nodes[parent_index]
      if child_index == parent_index:
        continue
      #　エッジ間のスコアが最大になる親ノードを探索
      candidate_model.add_edge(parent_node, child_node)
      current_score = self.metric.score(candidate_model)
      # 最大スコアのノード発見
      if current_score > max_score:
        max_score = current_score
        selected_parent_index = parent_index
        selected_parent_node = parent_node
      candidate_model.remove_edge(parent_node, child_node)
    return selected_parent_node, selected_parent_index

  def sample_data(self, new_data_size):
    inference = BayesianModelSampling(self.model)
    sampled_data = inference.forward_sample(size=new_data_size, return_type='dataframe')
    sampled_data = self.get_data_sorted_by_columns(sampled_data)
    return sampled_data.to_numpy()

  def is_dag(self, model, u, v):
    if u == v:
      return False
    if u in model.nodes() and v in model.nodes() and nx.has_path(model, v, u):
      return False
    return True

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
  N = 45
  POP_SIZE = 5000
  pop1 = Population(POP_SIZE, N)
  # pop1.print_population()
  for individual in pop1.array:
    individual.gene[2] = individual.gene[0] + individual.gene[1]
  BN = BayesianNetwork(pop1.array, 2)
  BN.construct_network_by_k2_algorithm()