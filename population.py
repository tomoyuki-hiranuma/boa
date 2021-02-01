import numpy as np
from individual import Individual

class Population:
  def __init__(self, population_size, individual_size):
    self.array = np.array([Individual(individual_size) for i in range(population_size) ])
    self.individual_size = individual_size
  
  def print_population(self):
    for individual in self.array:
      print(individual.gene)


if __name__ == '__main__':
  N = 6
  POPULATION_SIZE = 50

  pop1 = Population(POPULATION_SIZE, N)
  pop1.print_population()
