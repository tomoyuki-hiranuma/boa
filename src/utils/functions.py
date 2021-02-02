'''
  必要になった評価関数をクラスとして追加していく
  OneMax
  NK-model
  Trap-5
  3-Deceptiveなど
  遺伝子を受け取り評価値を返す関数calc_evaluation()を主に用いる
'''
from ..individual import Individual

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
    for index in range(0, len(individual.gene)-2, 3):
      print(index)
    return eval

if __name__ == '__main__':
  
  ind1 = Individual(5)
  ind1.print_info()