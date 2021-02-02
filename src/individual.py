import numpy as np
from copy import deepcopy
class Individual():
  def __init__(self, gene_size):
    self.gene = np.random.randint(2, size=gene_size)
    self.fitness = 0.0

  def set_gene(self, gene):
    self.gene = deepcopy(gene)

  def print_info(self):
    print("{} : {}".format(self.gene, self.fitness))

  def get_row(self):
    return [self.gene, self.fitness]
