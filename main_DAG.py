import run_DAG
import data_generator
import random
import numpy as np

import time

if __name__ == '__main__':
    alpha_l_list = [0.1]
    alpha_i_list = [0.4]
    # alpha_l_list = [0.01, 0.05, 0.1, 0.5]
    # alpha_i_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    # alpha_p = [0.01, 0.1, 1]
    alpha_p = [1]

    experiment_set = [(10, 9, 1)]
    # experiment_set = [(10, 9, 1), (5, 4, 1), (10, 10, 2), (10, 5, 1), (5, 5, 2)]
    sample_size = [100]
    # sample_size = [11, 20, 50, 100]
    population_size = 10000
    dag_size = 1000

    t = time.strftime("%Y-%m-%d", time.localtime())
    for e in experiment_set:
        print("========================")
        
        print(e)
        DAG_list, data_list, B_list = data_generator.generate_DAGs(e[0], e[1], e[2], population_size, dag_size)
        data_list = np.array(data_list)
        for s in sample_size:
            print("------------------------")
            indexs = np.random.randint(population_size, size=s)
            name = str(e[0]) + "_" + str(e[1]) + "_" + str(e[2]) + "_" + str(s) + "_" + t + "_looptest"
            x_list = data_list[:, indexs, :]
            run_DAG.run(name, DAG_list, x_list, B_list, alpha_l_list, alpha_i_list, alpha_p)

