'''
  必要になった評価関数をクラスとして追加していく
  OneMax
  NK-model
  Trap-5
  3-Deceptiveなど
  遺伝子を受け取り評価値を返す関数calc_evaluation()を主に用いる
'''
# from individual import Individual
import numpy as np


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
		self.best_eval = 0.0
		self.best_gene = ""

	def _create_NK_landscape(self):
		np.random.seed(1)
		index = [ f'{i:0{self.K+1}b}' for i in range(2**(self.K+1)) ]
		rand_array = np.random.random(2**(self.K+1))
		return dict(zip(index, rand_array))

	# 適応度を計算する
	def calc_eval(self, gene):
		fitness = 0.0
		long_genes = gene + gene
		for i in range(len(gene)):
			fitness += self.nk_landscape[long_genes[i:i+self.K+1]]
		fitness /= len(gene)
		return fitness

	# 最適解計算
	def calc_optimization(self):
		best_gene = ""
		best_eval = 0.0
		all_genes = np.array([ f'{i:0{self.N}b}' for i in range(2**(self.N)) ])
		for gene in all_genes:
			fitness = self.calc_eval(gene)
			if best_eval <= fitness:
				best_eval = fitness
				best_gene = gene
		self.best_eval = best_eval
		self.best_gene = best_gene

	@property
	def get_best_eval(self):
		return self.best_eval

	@property
	def get_best_gene(self):
		return self.best_gene

	def get_optimized_solution(self):
		return self.best_gene, self.best_eval


if __name__ == '__main__':
  N = 7
  ind1 = Individual(N)
  ind1.print_info()
  function = ThreeDeceptive()
  ind1.fitness = function.calc_evaluation(ind1)
  ind1.print_info()
  
