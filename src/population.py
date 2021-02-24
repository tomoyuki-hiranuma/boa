import numpy as np
from individual import Individual
import csv
from copy import deepcopy

class Population:
  def __init__(self, population_size, individual_size):
    self.array = np.array([Individual(individual_size) for i in range(population_size) ])
    self.individual_size = individual_size

  def copy(self):
    return deepcopy(self)

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

  def get_max_min_score(self):
    max_score = 0.0
    min_score = float('inf')
    for individual in self.array:
      if min_score > individual.fitness:
        min_score = individual.fitness
      if max_score < individual.fitness:
        max_score = individual.fitness
    return max_score, min_score

  def get_mean_variance_score(self):
    mean_score = self.get_mean_score()
    variance_score = 0.0
    for individual in self.array:
      variance_score += (individual.fitness - mean_score)**2
    variance_score /= len(self.array)
    return mean_score, variance_score

  def get_mean_score(self):
    mean_score = 0.0
    for individual in self.array:
      mean_score += individual.fitness
    return mean_score/len(self.array)

  def output_to_csv(self, file_name, gen):
    with open(file_name, 'a') as f:
      writer = csv.writer(f)
      for individual in self.array:
        writer.writerow([gen] + individual.get_row())
  
  def output_distribution_info(self, file_name, gen):
    max_score, min_score = self.get_max_min_score()
    mean_score, variance_score = self.get_mean_variance_score()
      
    with open(file_name, 'a') as f:
      writer = csv.writer(f)
      writer.writerow([gen, max_score, mean_score, min_score, variance_score])


if __name__ == '__main__':
  N = 6
  POPULATION_SIZE = 50

  pop1 = Population(POPULATION_SIZE, N)
  pop1.print_population()
