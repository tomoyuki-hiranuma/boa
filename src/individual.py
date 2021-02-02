import numpy as np

class Individual():
  def __init__(self, gene_size):
    np.random.seed(0)
    self.gene = np.random.randint(2, size=gene_size)
    self.fitness = 0.0

  def set_gene(self, gene):
    self.gene = gene

  def print_info(self):
    print("{} : {}".format(self.gene, self.fitness))
