import csv
import matplotlib.pyplot as plt
import japanize_matplotlib
import os
import datetime
import numpy as np

def get_eval_number(file_name, eval_number_per_gen):
  total_generation = 0

  with open(file_name, newline='') as f:
      reader = csv.reader(f)
      header = next(reader)

      for row in reader:
        if row[0] == "EOF":
          break
        total_generation += 1
  return eval_number_per_gen * total_generation

if __name__ == '__main__':
  N = 30
  POPULATION_SIZE = 1000
  eval_numbers_array = []

  for i in range(10):
    trial = i+1
    if trial == 9:
      continue

    file_name = "data/jsai/3-deceptive/N={}/POP={}/BOA_POP={}_N={}_trial{}.csv".format(N, POPULATION_SIZE, POPULATION_SIZE, N, trial)

    eval_number = get_eval_number(file_name, POPULATION_SIZE//2)
    eval_numbers_array.append(eval_number)
  eval_numbers_array = np.array(eval_numbers_array)
  print(eval_numbers_array.mean())
  var = eval_numbers_array.var()
  std = eval_numbers_array.std()
  print(std)

        
