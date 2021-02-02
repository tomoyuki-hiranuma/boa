import numpy as np

class Individual():
  def __init__(self, gene_size):
    self.gene = np.random.randint(2, size=gene_size)
    self.fitness = 0.0

  def print_info(self):
    print("{} : {}".format(self.gene, self.fitness))

if __name__ == '__main__':
  N = 6
  ind1 = Individual(N)
  print(ind1.gene)
  ind1.print_info()