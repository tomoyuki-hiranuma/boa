'''
  必要になった評価関数をクラスとして追加していく
  OneMax
  NK-model
  Trap-5
  3-Deceptiveなど
  遺伝子を受け取り評価値を返す関数calc_evaluation()を主に用いる
'''
# from individual import Individual

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
        eval += 0.9
      elif sum == 1:
        eval += 0.8
      elif sum == 2:
        eval += 0.0
      else:
        eval += 1.0
    return eval

if __name__ == '__main__':
  N = 7
  ind1 = Individual(N)
  ind1.print_info()
  function = ThreeDeceptive()
  ind1.fitness = function.calc_evaluation(ind1)
  ind1.print_info()
  
