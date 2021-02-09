import os
from src.population import Population
from src.bayesianNetwork import BayesianNetwork
from src.utils.functions import Onemax, ThreeDeceptive, NKModel
from copy import deepcopy
import csv

class Boa:
  def __init__(self, population_size, individual_size, select_size, new_data_size, K=-1):
    self.population_size = population_size # populationから取得可能→いらない
    self.individual_size = individual_size # populationから取得可能→いらない
    self.select_size = select_size
    self.new_data_size = new_data_size
    self.population = Population(population_size, individual_size)
    self.bayesianNetwork = None # BayesianNetwork(self.population.array)
    self.selected_population = None # いらなくなるかも
    self.function = NKModel(individual_size, K)
    if K != -1:
      self.function.calc_optimization()

  def do_one_generation(self):
    self.population = self.get_sorted_population()
    selected_array = deepcopy(self.population.array[0:self.select_size])
    # ネットワーク生成、推定
    self.create_network(selected_array)
    self.bayesianNetwork.fit()

    # サンプル取得
    sampled_array = self.bayesianNetwork.sample_data(self.new_data_size)

    # 入れ替え用集団生成
    new_population = Population(len(sampled_array), self.individual_size)
    new_population.set_array(sampled_array)

    # 入れ替える
    self.population.array[len(self.population.array) - self.new_data_size:len(self.population.array)] = new_population.array
    self.evaluate()
    self.sort_population()

  def get_sorted_population(self):
    sorted_population = deepcopy(self.population)
    sorted_population.array = sorted(sorted_population.array, key=lambda x: x.fitness)[::-1]
    return sorted_population

  def sort_population(self):
    self.population = self.get_sorted_population()

  def create_network(self, selected_array):
    self.bayesianNetwork = BayesianNetwork(selected_array)

  def evaluate(self):
    for individual in self.population.array:
      individual.fitness = self.function.calc_evaluation(individual)

  def get_mean_eval(self):
    eval = 0.0
    for individual in self.population.array:
      eval += individual.fitness
    return eval/len(self.population.array)
  
  def get_best_eval(self):
    return self.population.array[0].fitness

  def output_to_csv(self, file_name, generation):
    self.population.output_to_csv(file_name, generation)

  def is_convergence(self):
    return self.population.is_convergence()

if __name__ == '__main__':
  '''
    パラメータ設定
    POPULATION_SIZE: 集団サイズ(論文ではN)
    N: 遺伝子長(論文ではn)
    TAU: 切り捨て選択の割合 50%
    SELECT_SIZE: BN構築用に使われる個体群サイズ
    NEW_DATA_SIZE: BNから生成される個体群サイズ 下位個体群の半分が入れ変わる
  '''
  POPULATION_SIZEs = [500, 600, 700, 800, 900]
  for POPULATION_SIZE in POPULATION_SIZEs:
    N = 10
    K=0
    TAU = 0.5
    SELECT_SIZE = int(POPULATION_SIZE * (1.0 - TAU))
    NEW_DATA_SIZE = int(POPULATION_SIZE * TAU)
    MAX_EXPERIMENT = 30
    MAX_EVAL_NUM = 2000 * N
    
    FILE_NAME = "data/NK_model/N={}_K={}/BOA_POP={}_N={}_NKModel_K={}_new={}.csv".format(N, K, POPULATION_SIZE, N, K, NEW_DATA_SIZE)
    dir_name = FILE_NAME.split("/BOA")[0]

    
    boa = Boa(POPULATION_SIZE, N, SELECT_SIZE, NEW_DATA_SIZE, K)

    '''
      3-deceptiveのとき
        OPT_EVAL = N//3
      NK-Modelのとき
        OPT_EVAL = boa.function.get_best_eval
    '''
    OPT_EVAL = boa.function.get_best_eval
    opt_gene = boa.function.get_best_gene

    boa.evaluate()
    boa.sort_population()
    
    generation = 0
    eval_num = 0
    mean_eval = 0.0
    best_eval = 0.0
    is_converge = False
    optimal_rate = 1.0

    os.makedirs(dir_name, exist_ok=True)
    header = ['generation', 'individual', 'fitness']
    with open(FILE_NAME, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(header)
      writer.writerow(["opt", opt_gene, OPT_EVAL])
      writer.writerow(["START!"])

    boa.output_to_csv(FILE_NAME, generation)

    while eval_num < MAX_EVAL_NUM and best_eval < OPT_EVAL * optimal_rate and not is_converge:
      print("第{}世代".format(generation + 1))
      boa.do_one_generation()
      generation += 1
      eval_num += NEW_DATA_SIZE
      mean_eval = boa.get_mean_eval()
      best_eval = boa.get_best_eval()
      print("mean eval: {}".format(mean_eval))
      print("best eval: {}".format(best_eval))
      is_converge = boa.is_convergence()
      if generation%5 == 0 or is_converge or best_eval >= OPT_EVAL * optimal_rate:
        boa.output_to_csv(FILE_NAME, generation)

    boa.population.print_head_population()
    print("mean eval: {}".format(mean_eval))
    print("best eval: {}".format(best_eval))
    with open(FILE_NAME, 'a') as f:
      writer = csv.writer(f)
      writer.writerow(["EOF"])
      if is_converge:
        print("収束して失敗")
        writer.writerow(["fail"])
      elif best_eval >= OPT_EVAL * optimal_rate:
        print("成功")
        print(boa.bayesianNetwork.network.edges())
        writer.writerow(["success"])
      else:
        print("評価回数の限界値のため失敗")
        writer.writerow(["fail"])