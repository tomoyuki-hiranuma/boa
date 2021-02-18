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

  def estimate_network_by_k2(self):
    self.network = self.construct_network_by_k2_algorithm()

  def estimate_network_by_hillclimb(self):
    estimater = HillClimbSearch(self.data)
    best_model = estimater.estimate(scoring_method=BicScore(self.data))
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
      【フロー】
      全てのノード間で最もBICスコアの高いノード間にエッジを張る（このときn個のノード、1つのエッジ）
      エッジを保存した状態で全てのノード間のBICスコアを計算し、最も高いノード間にエッジを張る(このときn個のノード、2つのエッジ)
      最終的にスコアが負になったら探索終了
    '''
  def construct_network_by_k2_algorithm(self):
    network = []
    masks_table = np.eye(len(self.nodes), dtype=bool)
    for child_index in range(len(self.nodes)):
      current_model = BayesianModel(network)
      current_model.add_nodes_from(self.nodes)
      ok_to_proceed = True
      child_node = self.nodes[child_index]
      while ok_to_proceed and len(list(current_model.predecessors(child_node))) < self.max_indegree:
        # 最大となる親ノード候補を抽出
        max_score = -float('inf')
        selected_parent_node = None
        for diff_index in range(len(self.nodes[child_index:])):
          parent_index = child_index + diff_index
          parent_node = self.nodes[parent_index]
          if child_index == parent_index:
            continue
          if masks_table[child_index, parent_index] or not self.is_dag(current_model, parent_node, child_node):
            continue

          network_candidate = BayesianModel(network)
          network_candidate.add_nodes_from(self.nodes)
          network_candidate.add_edge(parent_node, child_node)
          current_score = self.get_k2_score(network_candidate)
          # 最大スコアのノード発見
          if current_score > max_score:
            max_score = current_score
            selected_parent_index = parent_index
            selected_parent_node = parent_node

        # 見つからなかった
        if selected_parent_node == None:
          break
        
        # print("candidate parent: ", selected_parent_node)
        new_model = BayesianModel(network)
        new_model.add_nodes_from(self.nodes)
        new_model.add_edge(selected_parent_node, child_node)
        if self.get_k2_score(new_model) > self.get_k2_score(current_model):
          network.append([selected_parent_node, child_node])
          current_model.add_edge(selected_parent_node, child_node)
          masks_table[selected_parent_index, child_index] = True
          masks_table[child_index, selected_parent_index] = True
        else:
          ok_to_proceed = False
        print("network:",network)
    return network

  def is_proceed_ok(self, network, score_table, added_nodes_index):
    parent_node = "X"+str(added_nodes_index[0]+1)
    child_node = "X"+str(added_nodes_index[1]+1)

    current_model = BayesianModel(network)
    current_model.add_nodes_from(self.nodes)
    candidate_model = BayesianModel(network)
    candidate_model.add_nodes_from(self.nodes)
    candidate_model.add_edge(parent_node, child_node)

    if self.get_k2_score(candidate_model) >= self.get_k2_score(current_model):
      return True
    return False

  def get_bic_score(self, model):
    score = 0
    for node in model.nodes():
      score += self.local_bic_score(node, model.predecessors(node))
    return score

  def get_k2_score(self, model):
    k2 = K2Score(self.data)
    return k2.score(model)

  '''
  todo: 論文のBICに書き換える必要あり
  だいぶ付け焼き刃
  自作可能
  '''
  def local_bic_score(self, variable, parents):
    bic = BicScore(self.data)
    var_state = bic.state_names[variable] # 着目ノードが何の値を取るのか
    var_cardinality = len(var_state) # 着目ノードの取れる値の数、すなわち濃度
    state_counts = bic.state_counts(variable, parents) # 親ノードに対する着目ノードのデータの数
    sample_size = len(self.data)  # 総データ数
    num_parents_states = float(state_counts.shape[1]) # 親ノードの出力の組み合わせ数
    number_of_parent = self.get_parent_number(parents)

    counts = np.asarray(state_counts) # state_countsをnp.arrayにする
    log_likelihoods = np.zeros_like(counts, dtype=np.float_) # countsと同様のshapeの0の配列を生成
    # countsの自然対数を取る.出力形式はlog_likelihoodsの形で正の値に対してのみ対数を取る
    np.log(counts, out=log_likelihoods, where=counts > 0)

    log_conditionals = np.sum(counts, axis=0, dtype=np.float_) # countsを縦に総和
    np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)

    log_likelihoods -= log_conditionals
    log_likelihoods *= counts

    score = np.sum(log_likelihoods)
    score -= 0.5 * log(sample_size) * num_parents_states * (var_cardinality - 1)
    # entropy = np.sum(log_likelihoods)
    # score = - entropy * sample_size - 2 ** (number_of_parent) *  log2(sample_size) * 0.5
    return score

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

  def get_parent_number(self, parents):
    number = 0
    for parent in parents:
      number += 1
    return number

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
  BN = BayesianNetwork(pop1.array, 2)
  BN.construct_network_by_k2_algorithm()
  # print(BN.data)
  # BN.estimate()
  # BN.fit()
  # cpds = BN.model.get_cpds()
  # for cpd in cpds:
  #   print(cpd)
  # new_data = BN.sample_data(new_data_size=20)
  # print(new_data)