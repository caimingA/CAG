import os
import pandas as pd
import numpy as np
import re
import xlwt

# alpha_l_list = [0.01, 0.05, 0.1, 0.5]
# alpha_i_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
alpha_l_list = [0.01, 0.05, 0.1, 0.5]
alpha_i_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]


e_dict = dict()
count = 0
for i in alpha_l_list:
    for j in alpha_i_list:
        e_dict[count] = [i, j]
        count += 1

def read_one_excel(excel_name):
    # print(excel_name)
    name_list = re.split('_|\\\\', excel_name)
    # print(name_list)
    df_1 = pd.read_excel(excel_name, sheet_name="proposed", header=None)
    df_2 = pd.read_excel(excel_name, sheet_name="comp", header=None)
    df_3 = pd.read_excel(excel_name, sheet_name="RCD", header=None)

    # print(df_1)
    # print(df_2)
    index = df_1[2].idxmax()
    # print(index)
    res_1 = np.array(df_1.iloc[index,:]).tolist()
    res_1 = np.array([int(name_list[3]), int(name_list[4]), int(name_list[5]), int(name_list[6])] + e_dict[index] + res_1)
    # print(res_1)
    res_2 = np.array(df_2).tolist()
    temp_1 = np.array([int(name_list[3]), int(name_list[4]), int(name_list[5]), int(name_list[6])] + [0, 0] + res_2[0])
    temp_2 = np.array([int(name_list[3]), int(name_list[4]), int(name_list[5]), int(name_list[6])] + [0, 0] + res_2[1])
    
    index = df_3[2].idxmax()
    # print(index)
    res_3 = np.array(df_3.iloc[index,:]).tolist()
    res_3 = np.array([int(name_list[3]), int(name_list[4]), int(name_list[5]), int(name_list[6])] + e_dict[index] + res_3)
    
    
    res = np.array([res_1, temp_1, temp_2, res_3])
    # print(res)
    return res

def read_excel_dir(dir_path, fake_files):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet('data',cell_overwrite_ok=True) 
    for parents, dirnames, filenames in os.walk(dir_path):
        count = 0
    
        for excel_name in fake_files:
            if excel_name not in filenames:
                continue
            print(excel_name)
            data = read_one_excel(dir_path + str("\\") + excel_name)
            # print(data)
            for i in range(len(data)):
                for j in range(len(data[0])):
                    sheet1.write(count, j , data[i][j])
                count += 1
        # for excel_name in filenames:
        #     print(excel_name)
        #     if excel_name == "desktop.ini":
        #         continue
        #     data = read_one_excel(dir_path + str("\\") + excel_name)
        #     # print(data)
        #     for i in range(len(data)):
        #         for j in range(len(data[0])):
        #             sheet1.write(count, j , data[i][j])
        #         count += 1
    
    f.save(dir_path + r"\result" + ".xls")


if __name__ == '__main__':
    experiment_set = [(10, 5, 1), (10, 9, 1), (20, 19, 1), (30, 29, 1), (40, 39, 1)]
    sample_size = [11, 25, 50, 100, 500, 1000]
    # res = read_one_excel("10_9_1_100_2023-04-20.xls")
    # print(res)
    # dir_path = r".\one_1"
    # dir_path = r".\one_2"
    # dir_path = r".\two_1"
    # dir_path = r".\two_2"
    # dir_path = r".\three_1"
    # dir_path = r".\three_2"
    # dir_path = r".\four_1"
    # dir_path = r".\four_2"
    # dir_path = r".\plus_plus"
    # dir_path = r".\0707_original"
    dir_path = r".\1021_wald"
    
    # post = ".xls"
    post = "_Wald.xls"
    # post = "_Ance.xls"
    # post = "_Valu.xls"
    time = "2023-10-21"
    file_name = list()
    for e in experiment_set:
        for s in sample_size:
            item = str(e[0]) + "_" + str(e[1]) + "_" + str(e[2]) + "_" + str(s) + "_" + time + post
            file_name.append(item)
    # print(file_name)
    
    read_excel_dir(dir_path, file_name)