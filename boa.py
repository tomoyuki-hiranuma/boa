from population import Population
from bayesianNetwork import BayesianNetwork
from utils.functions import Onemax
from copy import deepcopy

class Boa:
  def __init__(self, population_size, individual_size, select_size, change_size):
    self.population_size = population_size # populationから取得可能→いらない
    self.individual_size = individual_size # populationから取得可能→いらない
    self.select_size = select_size
    self.change_size = change_size
    self.population = Population(population_size, individual_size)
    self.bayesianNetwork = BayesianNetwork(self.population.array)
    self.selected_population = None # いらなくなるかも
    self.function = Onemax()

  def do_one_generation(self):
    self.evaluate()
    self.population = self.get_sorted_population()

  def get_sorted_population(self):
    sorted_population = deepcopy(self.population)
    sorted_population.array = sorted(sorted_population.array, key=lambda x: x.fitness)[::-1]
    return sorted_population

  # def select_population(self):

  def evaluate(self):
    for individual in self.population.array:
      individual.fitness = self.function.calc_evaluation(individual)


if __name__ == '__main__':
  POPULATION_SIZE = 4
  N = 6
  SELECT_SIZE = 3
  CHANGE_SIZE = 5

  boa = Boa(POPULATION_SIZE, N, SELECT_SIZE, CHANGE_SIZE)
  boa.do_one_generation()
  boa.population.print_population()