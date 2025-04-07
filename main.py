import run
import data_generator
import random
import os
import numpy as np
import pandas as pd
import re

import time


def read_one_excel(excel_name):
    res = list()
    df = pd.read_excel(excel_name, sheet_name=None, header=None)

    # print(len(df))
    # print(df.keys())
    key_list = df.keys()
    for key in key_list:
        res.append(df[key].values)
    # df_1 = pd.read_excel(excel_name, sheet_name="DAG", header=None)
    # df_2 = pd.read_excel(excel_name, sheet_name="B", header=None)
    return res

def read_excel_dir(dir_path):
    res_list = list()
    print(dir_path)
    for parents, dirnames, filenames in os.walk(dir_path):
        count = 0
        for excel_name in filenames:
            res = read_one_excel(dir_path + "/" + excel_name)
            # print(excel_name)
            res_list.append(res)
    return res_list


if __name__ == '__main__':
    # alpha_l_list = [0.01]
    # alpha_i_list = [0.001]
    alpha_l_list = [0.01, 0.05, 0.1, 0.5]
    alpha_i_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    # alpha_i_list = [0.001, 0.01, 0.1]
    # alpha_p = [0.01, 0.1, 1]
    alpha_p = [1]

    # experiment_set = [(10, 5, 1)]
    # experiment_set = [(10, 9, 1)]
    # experiment_set = [(20, 19, 1)]
    # experiment_set = [(30, 29, 1)]
    # experiment_set = [(40, 39, 1)]
    experiment_set = [(10, 5, 1), (10, 9, 1), (10, 10, 2), (10, 12, 2), (10, 14, 2), (10, 16, 2), (10, 17, 2), (10, 24, 3)]
    # experiment_set = [(20, 19, 1)]
    # experiment_set = [(30, 29, 1)]
    # experiment_set = [(40, 39, 1)]
    # sample_size = [11, 25, 50, 100, 500]
    # sample_size = [50, 100, 1000]
    # sample_size = [11]
    # sample_size = [1000]
    population_size = 10000
    dag_size = 100

    # dir_list = [r".\10_5_1_DAG", r".\10_9_1_DAG"]
    dir_list = list()
    for e in experiment_set:
        dir_path = "./" + str(e[0]) + "_" + str(e[1]) + "_" + str(e[2]) + "_" + "DAG"

        dir_list.append(dir_path) 
    t = time.strftime("%Y-%m-%d", time.localtime())

    count = 0
    for dir_path in dir_list:
        print(dir_path)
        print("========================")
        e = experiment_set[count]
        print(e)

        res_list = read_excel_dir(dir_path)
        print(len(res_list))
        # print(len(res_list[0]))
        DAG_list = list()
        B_list = list() 
        
        for i in range(dag_size):
            # print(i)
            DAG_list.append(res_list[i][0])
            B_list.append(res_list[i][1])
        # for i in range(2, len(res_list)):
        DAG_list = np.array(DAG_list)
        B_list = np.array(B_list)
        for s in range(2, len(res_list[0])):
            if len(res_list[0][s]) <= 500:
                print("-------" + str(len(res_list[0][s])) + "-------")
                name = str(e[0]) + "_" + str(e[1]) + "_" + str(e[2]) + "_" + str(len(res_list[0][s])) + "_" + t
                # print(name)
                x_list = list()
                for j in range(dag_size):
                    x_list.append(res_list[j][s])
                x_list = np.array(x_list)
                # print(DAG_list.shape, " + ", B_list.shape, " + ", x_list.shape)
                run.run(name, DAG_list, x_list, B_list, alpha_l_list, alpha_i_list, alpha_p)
        count += 1





####### 2023/10/18 #######
# if __name__ == '__main__':
#     # alpha_l_list = [0.5]
#     # alpha_i_list = [0.5]
#     alpha_l_list = [0.01, 0.05, 0.1, 0.5]
#     alpha_i_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
#     # alpha_p = [0.01, 0.1, 1]
#     alpha_p = [1]

#     # experiment_set = [(40, 39, 1)]
#     # experiment_set = [(5, 4, 1)]
#     # experiment_set = [(10, 9, 1), (5, 4, 1), (10, 5, 1)]
#     # experiment_set = [(10, 5, 1), (10, 9, 1), (20, 19, 1)]
#     experiment_set = [(10, 5, 1)]
#     # experiment_set = [(30, 29, 1), (40, 39, 1)]
#     # experiment_set = [(40, 39, 1)]
#     # experiment_set = [(10, 9, 1), (5, 4, 1), (10, 10, 2), (10, 5, 1), (5, 5, 2)]
#     # sample_size = [50, 100, 1000]
#     # sample_size = [50, 100, 1000]
#     # sample_size = [11]
#     sample_size = [11]
#     population_size = 10000
#     dag_size = 20



#     t = time.strftime("%Y-%m-%d", time.localtime())
#     for e in experiment_set:
#         print("========================")
        
#         print(e)
#         DAG_list, data_list, B_list = data_generator.generate_DAGs(e[0], e[1], e[2], population_size, dag_size)
#         data_list = np.array(data_list)
#         # print(data_list[0].shape)
#         for s in sample_size:
#             if e[0] < s:
#                 print("------------------------")
#                 indexs = np.random.randint(population_size, size=s)
#                 name = str(e[0]) + "_" + str(e[1]) + "_" + str(e[2]) + "_" + str(s) + "_" + t
#                 x_list = data_list[:, indexs, :]

#                 # print(x_list.shape)
#                 run.run(name, DAG_list, x_list, B_list, alpha_l_list, alpha_i_list, alpha_p)
