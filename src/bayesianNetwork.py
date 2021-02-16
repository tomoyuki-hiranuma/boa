import pandas as pd
import numpy as np
from pgmpy.models import BayesianModel
from copy import deepcopy
from math import log
from pgmpy.estimators import K2Score, BicScore, HillClimbSearch, ExhaustiveSearch
from pgmpy.sampling import BayesianModelSampling
from population import Population

class BayesianNetwork:
  def __init__(self, individual_array):
    self.network = [] # 
    self.model = None # BayesianModelクラスのインスタ
    self.nodes = None # nodeのリスト
    self.data = self._to_DataFrame(individual_array) # pandas.DataFrame

  def estimate_network_by_k2(self):
    self.network = self.construct_network_by_k2_algorithm()

  def estimate_network_by_hillclimb(self):
    estimater = HillClimbSearch(self.data)
    best_model = estimater.estimate(scoring_method=BicScore(self.data))
    self.network = list(best_model.edges())

  def fit(self):
    print("ネットワーク構築")
    self.estimate_network_by_hillclimb()
    # 構築したネットワークのエッジを使う
    self.model = BayesianModel(self.network)
    # 独立なノードがあったとき
    self.model.add_nodes_from(self.nodes)
    self.model.fit(self.data)

  def construct_network_by_k2_algorithm(self):
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
    for _ in range(5): ## while True にしてスコアが正になるまで
      scores_table = np.ones((len(self.nodes), len(self.nodes)))*(-float('inf'))
      current_model = BayesianModel(network)
      print("{}回目".format(_+1))
      print("edge: {}".format(network))
      print("score: {}".format(self.get_bic_score(current_model))) # エントロピーが加算されてない
      # テーブル再計算
      for parent_index, parent_node in enumerate(self.nodes):
        for child_index, child_node in enumerate(self.nodes):
          if not masks_table[parent_index, child_index]:
            network_candidate = BayesianModel(network)
            network_candidate.add_nodes_from(self.nodes)
            network_candidate.add_edge(parent_node, child_node)
            scores_table[parent_index, child_index] = self.get_bic_score(network_candidate)
      '''
        スコアが改善するなら改善させる
        しないなら終了
      '''
      # if np.max(scores_table) < 0:
        # break
      print(scores_table)
      nodes_index = np.unravel_index(np.argmax(scores_table), scores_table.shape)
      if not masks_table[nodes_index[0], nodes_index[1]]:
        network.append(["X"+str(nodes_index[0]+1), "X"+str(nodes_index[1]+1)])
        masks_table[nodes_index[0], nodes_index[1]] = True
        masks_table[nodes_index[1], nodes_index[0]] = True
      print("結果:{}".format(network))
    return network

  def get_bic_score(self, model):
    score = 0
    for node in model.nodes():
      score += self.local_bic_score(node, model.predecessors(node))
    return score

  '''
  todo: 論文のBICに書き換える必要あり
  だいぶ付け焼き刃
  '''
  def local_bic_score(self, variable, parents):
    bic = BicScore(self.data)
    var_state = bic.state_names[variable] # 着目ノードが何の値を取るのか
    var_cardinality = len(var_state) # 着目ノードの取れる値の数、すなわち濃度
    state_counts = bic.state_counts(variable, parents) # 親ノードに対する着目ノードのデータの数
    sample_size = len(self.data)  # 総データ数
    num_parents_states = float(state_counts.shape[1]) # 親ノードの出力の組み合わせ数

    counts = np.asarray(state_counts) # state_countsをnp.arrayにする
    log_likelihoods = np.zeros_like(counts, dtype=np.float_) # countsと同様のshapeの0の配列を生成
    # countsの自然対数を取る.出力形式はlog_likelihoodsの形で正の値に対してのみ対数を取る
    counts = np.log(counts, out=log_likelihoods, where=counts > 0)

    log_conditionals = np.sum(counts, axis=0, dtype=np.float_) # countsを縦に総和
    log_conditionals = np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)

    log_likelihoods -= log_conditionals
    log_likelihoods *= counts
    score = np.sum(log_likelihoods)
    score -= 0.5 * log(sample_size) * num_parents_states * (var_cardinality - 1)
    return score

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
  BN.construct_network_by_k2_algorithm()
  # print(BN.data)
  # BN.estimate()
  # BN.fit()
  # cpds = BN.model.get_cpds()
  # for cpd in cpds:
  #   print(cpd)
  # new_data = BN.sample_data(new_data_size=20)
  # print(new_data)