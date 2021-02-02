from population import Population
from bayesianNetwork import BayesianNetwork
from utils.functions import Onemax
from copy import deepcopy

class Boa:
  def __init__(self, population_size, individual_size, select_size, new_data_size):
    self.population_size = population_size # populationから取得可能→いらない
    self.individual_size = individual_size # populationから取得可能→いらない
    self.select_size = select_size
    self.new_data_size = new_data_size
    self.population = Population(population_size, individual_size)
    self.bayesianNetwork = None # BayesianNetwork(self.population.array)
    self.selected_population = None # いらなくなるかも
    self.function = Onemax()

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
    

  def get_sorted_population(self):
    sorted_population = deepcopy(self.population)
    sorted_population.array = sorted(sorted_population.array, key=lambda x: x.fitness)[::-1]
    return sorted_population

  def create_network(self, selected_array):
    self.bayesianNetwork = BayesianNetwork(selected_array)

  def evaluate(self):
    for individual in self.population.array:
      individual.fitness = self.function.calc_evaluation(individual)


if __name__ == '__main__':
  POPULATION_SIZE = 30
  N = 6
  SELECT_SIZE = 10
  NEW_DATA_SIZE = 5
  GENERATIONS = 10

  boa = Boa(POPULATION_SIZE, N, SELECT_SIZE, NEW_DATA_SIZE)
  boa.evaluate()
  for i in range(GENERATIONS):
    boa.do_one_generation()

  boa.population.print_population()