import numpy as np
from src.individual import Individual

class Population:
  def __init__(self, population_size, individual_size):
    self.array = np.array([Individual(individual_size) for i in range(population_size) ])
    self.individual_size = individual_size

  def set_array(self, array):
    self.array = np.array([Individual(self.individual_size) for gene in array])
    for index, individual in enumerate(self.array):
      individual.set_gene(array[index])
  
  def print_population(self):
    for individual in self.array:
      individual.print_info()


if __name__ == '__main__':
  N = 6
  POPULATION_SIZE = 50

  pop1 = Population(POPULATION_SIZE, N)
  pop1.print_population()
