'''
  必要になった評価関数をクラスとして追加していく
  OneMax
  NK-model
  Trap-5など
  遺伝子を受け取り評価値を返す関数calc_evaluation()を主に用いる
'''

class Onemax:
  def __init__(self):
    pass

  def calc_evaluation(self, individual):
    return sum(individual.gene)