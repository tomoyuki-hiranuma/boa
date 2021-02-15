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
    self.nodes = None
    self.data = self._to_DataFrame(individual_array)

  def construct_network_by_k2_algorithm(self):
    BIC_tables = self.create_bic_tables()
    # エッジのみの配列として表す(親，子)
    network = []
    bic = BicScore(self.data)
    '''
      tableの上から順にノードとして追加
      貪欲法でスコア順に追加しつつ、すべての遺伝子組のスコアが負になるまでネットワーク構築
      【フロー】
      全てのノード間で最もスコアの高いノード間にエッジを張る（n個のノード、1つのエッジ）
      エッジを保存した状態で全てのノード間のスコアを計算し、最も高いノード間にエッジを張る(n個のノード、2つのエッジ)
      最終的にスコアが負になったら探索終了
    '''
    for parent_node, child_node, score in BIC_tables:
      # print(parent_node, child_node, score)
      network.append([parent_node, child_node])
      model = BayesianModel(network)
      print(model.edges())
      print(bic.score(model))
      # if bic.score(model) < 0:
        # break
    return network

  def estimate(self):
    self.network = self.construct_network_by_k2_algorithm()

  def fit(self):
    self.estimate()
    # 構築したネットワークのエッジを使う
    self.model = BayesianModel(self.network)
    # 独立なノードがあったとき
    self.model.add_nodes_from(self.nodes)
    self.model.fit(self.data)

  '''
    [親ノード, 子ノード, BICスコア]の配列
  '''
  def create_bic_tables(self):
    ## 前処理で(ペア, スコア)のテーブルを作る
    ## self.dataに対して、全ての組み合わせのBICを計算
    tables = []
    bic = BicScore(self.data)
    for par_label, par_items in self.data.iteritems():
      for child_label, child_items in self.data.iteritems():
        if par_label != child_label:
          # print("local score")
          # print("parent: {}, child: {}".format(par_label, child_label))
          # print(bic.local_score(child_label, [par_label]))
          tables.append((par_label, child_label, bic.local_score(child_label, [par_label])))
          # print("-----------\n")
    sorted_tables = sorted(tables, key=lambda x: x[2])[::-1]
    return np.array(sorted_tables)

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
  N = 3
  POP_SIZE = 50
  pop1 = Population(POP_SIZE, N)
  # pop1.print_population()
  BN = BayesianNetwork(pop1.array)
  bic_tables = BN.construct_network_by_k2_algorithm()
  # print(BN.data)
  # BN.estimate()
  # BN.fit()
  # cpds = BN.model.get_cpds()
  # for cpd in cpds:
  #   print(cpd)
  # new_data = BN.sample_data(new_data_size=20)
  # print(new_data)