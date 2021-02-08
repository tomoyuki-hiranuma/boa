import numpy as np
from src.individual import Individual
import csv

class Population:
  def __init__(self, population_size, individual_size):
    self.array = np.array([Individual(individual_size) for i in range(population_size) ])
    self.individual_size = individual_size

  def set_array(self, array):
    self.array = np.array([Individual(self.individual_size) for gene in array])
    for index, individual in enumerate(self.array):
      individual.set_gene(array[index])

  def is_convergence(self):
    array = np.array([individual.gene for individual in self.array])
    means = array.mean(axis=0)
    return (means < 0.03).all() or (means > 0.97).all()
  
  def print_population(self):
    for individual in self.array:
      individual.print_info()

  def print_head_population(self, number=5):
    for individual in self.array[0:number]:
      individual.print_info()

  def output_to_csv(self, file_name, gen):
    with open(file_name, 'a') as f:
      writer = csv.writer(f)
      for individual in self.array:
        writer.writerow([gen] + individual.get_row())

if __name__ == '__main__':
  N = 6
  POPULATION_SIZE = 50

  pop1 = Population(POPULATION_SIZE, N)
  pop1.print_population()
