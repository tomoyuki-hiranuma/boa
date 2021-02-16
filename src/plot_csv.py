import csv
import matplotlib.pyplot as plt
import japanize_matplotlib
import os

if __name__ == '__main__':
  N = 20
  Ks = range(0, N, 3)
  POPULATION_SIZEs = range(500, 1000, 100)

  opt_eval = 0.0
  for K in Ks:
    max_eval_num = 0
    for POPULATION_SIZE in POPULATION_SIZEs:
      file_name = "data/NK_model/N={}_K={}/BOA_POP={}_N={}_NKModel_K={}_new={}.csv".format(N, K, POPULATION_SIZE, N, K, POPULATION_SIZE//2)
      # 評価回数に対する集団の最良評価値をみてみる
      # 論文では最適解が求まる時の次元に対する集団サイズと評価回数
      eval_numbers = []
      best_evals = []
      with open(file_name, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        opt_solution = next(reader)
        opt_eval = float(opt_solution[2])
        header = next(reader)
        now_generation = -1
        eval_num = 0
        for row in reader:
          if row[0] == "EOF":
            break
          if now_generation != int(row[0]):
            eval_num += (int(row[0]) - now_generation) * POPULATION_SIZE / 2
            now_generation = int(row[0])
            eval_numbers.append(eval_num)
            best_evals.append(float(row[2]))
        if max_eval_num <= eval_num:
          max_eval_num = eval_num
      
      plt.plot(eval_numbers, best_evals, label="Pop size: {}".format(POPULATION_SIZE))
    plt.plot([0, max_eval_num], [opt_eval, opt_eval], label="opt solution", color="red", linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid()
    plt.xlabel("評価回数")
    plt.ylabel("最良評価値")
    plt.title("遺伝子サイズ{}のときのBOAにおける評価回数に対する最良評価値の推移".format(N))

    dir_name = "plot/NK_model/N={}_K={}/".format(N, K)
    os.makedirs(dir_name, exist_ok=True)
    plt.savefig(dir_name + "conpare_best_evals_by_eval_number_no_opt_line.png")
    plt.clf()