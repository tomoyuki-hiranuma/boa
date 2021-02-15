import pandas as pd
import numpy as np
from pgmpy.models import BayesianModel
from copy import deepcopy
from pgmpy.estimators import K2Score, BicScore, HillClimbSearch, ExhaustiveSearch
from pgmpy.sampling import BayesianModelSampling
from src.population import Population

class BayesianNetwork:
  def __init__(self, individual_array):
    self.network = []
    self.model = None
    self.nodes = None
    self.data = self._to_DataFrame(individual_array)

  # def construct_network_by_k2_algorithm(self):
  #   BIC_tables = self.create_bic_tables()
  #   # エッジのみの配列として表す(親，子)
  #   network = []
  #   bic = BicScore(self.data)
  #   '''
  #     tableの上から順にノードとして追加
  #     貪欲法でスコア順に追加しつつ、すべての遺伝子組のスコアが負になるまでネットワーク構築
  #     【フロー】
  #     全てのノード間で最もBICスコアの高いノード間にエッジを張る（このときn個のノード、1つのエッジ）
  #     エッジを保存した状態で全てのノード間のBICスコアを計算し、最も高いノード間にエッジを張る(このときn個のノード、2つのエッジ)
  #     最終的にスコアが負になったら探索終了
  #   '''
  #   for parent_node, child_node, score in BIC_tables:
  #     # print(parent_node, child_node, score)
  #     network.append([parent_node, child_node])
  #     model = BayesianModel(network)
  #     print(model.edges())
  #     print(bic.score(model))
  #     # if bic.score(model) < 0:
  #       # break
  #   return network

  def estimate_network(self):
    self.network = self.construct_network_by_k2_algorithm()

  def fit(self):
    self.estimate_network()
    # 構築したネットワークのエッジを使う
    self.model = BayesianModel(self.network)
    # 独立なノードがあったとき
    self.model.add_nodes_from(self.nodes)
    self.model.fit(self.data)

  '''
    [親ノード, 子ノード, BICスコア]の配列
  '''
  def create_bic_tables(self):
    '''
      K2アルゴリズム
      【フロー】
      全てのノード間で最もBICスコアの高いノード間にエッジを張る（このときn個のノード、1つのエッジ）
      エッジを保存した状態で全てのノード間のBICスコアを計算し、最も高いノード間にエッジを張る(このときn個のノード、2つのエッジ)
      最終的にスコアが負になったら探索終了
    '''
    network = []
    ## self.dataに対して、今のネットワークにおいて全ての組み合わせのBICを計算
    scores_table = np.ones((len(self.nodes), len(self.nodes)))*(-float('inf'))
    masks_table = np.eye(len(self.nodes), dtype=bool)
    bic = BicScore(self.data)
    for _ in range(3):
      scores_table = np.ones((len(self.nodes), len(self.nodes)))*(-float('inf'))
      print("{}回目".format(_+1))
      print(network)
      for parent_index, parent_node in enumerate(self.nodes):
        for child_index, child_node in enumerate(self.nodes):
          if not masks_table[parent_index, child_index]:
            network_candidate = BayesianModel(network)
            network_candidate.add_nodes_from(self.nodes)
            network_candidate.add_edge(parent_node, child_node)
            scores_table[parent_index, child_index] = bic.score(network_candidate)
      # if np.max(scores_table) < 0:
        # break
      print(scores_table)
      nodes_index = np.unravel_index(np.argmax(scores_table), scores_table.shape)
      network.append(["X"+str(nodes_index[0]+1), "X"+str(nodes_index[1]+1)])
      masks_table[nodes_index[0], nodes_index[1]] = True
      masks_table[nodes_index[1], nodes_index[0]] = True
      print("結果:{}".format(network))
    # return np.array(sorted_tables)

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
  for individual in pop1.array:
    individual.gene[2] = individual.gene[0] + individual.gene[1]
  BN = BayesianNetwork(pop1.array)
  BN.create_bic_tables()
  # print(BN.data)
  # BN.estimate()
  # BN.fit()
  # cpds = BN.model.get_cpds()
  # for cpd in cpds:
  #   print(cpd)
  # new_data = BN.sample_data(new_data_size=20)
  # print(new_data)