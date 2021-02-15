'''
  必要になった評価関数をクラスとして追加していく
  OneMax
  NK-model
  Trap-5
  3-Deceptiveなど
  遺伝子を受け取り評価値を返す関数calc_evaluation()を主に用いる
'''
import numpy as np
from copy import deepcopy
from src.individual import Individual

class Onemax:
  def __init__(self):
    pass

  def calc_evaluation(self, individual):
    return sum(individual.gene)

class ThreeDeceptive:
  def __init__(self):
    pass

  def calc_evaluation(self, individual):
    eval = 0.0
    for index in range(0, len(individual.gene), 3):
      sum = 0
      if index < len(individual.gene)-2:
        sum = individual.gene[index] + individual.gene[index+1] + individual.gene[index+2]
      elif index < len(individual.gene)-1:
        sum = individual.gene[index] + individual.gene[index+1]
      else:
        sum = individual.gene[index]
      
      if sum == 0:
        eval += 9
      elif sum == 1:
        eval += 8
      elif sum == 2:
        eval += 0
      else:
        eval += 10
    return eval/10

class NKModel:
  def __init__(self, N ,K):
    self.N = N
    self.K = K
    self.nk_landscape = self._create_NK_landscape()
    self.individual = Individual(N)
    self.calc_optimization()

  def _create_NK_landscape(self):
    np.random.seed(1)
    index = [ f'{i:0{self.K+1}b}' for i in range(2**(self.K+1)) ]
    rand_array = np.random.random(2**(self.K+1))
    return dict(zip(index, rand_array))

	# 適応度を計算する
  def calc_evaluation(self, individual):
    fitness = 0.0
    long_genes = np.hstack((individual.gene, individual.gene))
    for i in range(len(individual.gene)):
      elements = ""
      for j in long_genes[i:i+self.K+1]:
        elements += str(j)
      fitness += self.nk_landscape[elements]
    fitness /= len(individual.gene)
    return fitness

	# 最適解計算
  '''
    メモ化で高速化できそう
  '''
  def calc_optimization(self):
    best_gene = ""
    best_eval = 0.0
    all_genes = np.array([ f'{i:0{self.N}b}' for i in range(2**(self.N)) ])
    all_genes = self._to_np_int(all_genes)
    for gene in all_genes:
      individual = Individual(self.N)
      individual.gene = deepcopy(gene)
      fitness = self.calc_evaluation(individual)
      if best_eval <= fitness:
        best_eval = fitness
        best_gene = gene
    self.individual.fitness = best_eval
    self.individual.gene = best_gene

  def _to_np_int(self, binary_array):
    new_genes = []
    for gene in binary_array:
      gene = list(gene)
      genes = []
      for num in gene:
        genes.append(int(num))
      new_genes.append(np.array(genes))
    return np.array(new_genes)

  @property
  def get_best_eval(self):
    return self.individual.fitness

  @property
  def get_best_gene(self):
    return self.individual.gene

  def get_optimized_solution(self):
    return self.individual


if __name__ == '__main__':
  N = 7
  ind1 = Individual(N)
  ind1.print_info()
  function = ThreeDeceptive()
  ind1.fitness = function.calc_evaluation(ind1)
  ind1.print_info()
  
