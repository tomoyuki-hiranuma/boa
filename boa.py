from population import Population

class Boa:
  def __init__(self, population_size, individual_size):
    self.population_size = population_size
    self.individual_size = individual_size
    self.population = Population(population_size, individual_size)
