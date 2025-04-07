# import run
# import data_generator
# import random
# import os
import numpy as np
import pandas as pd
import draw
import execute_refined as ex

import evaluation
# import re

import time
# import pgmpy

import bnlearn as bn
import xlwt

# from IPython.display import Image
# from pgmpy.utils import get_example_model

# # Load the model
# asia_model = get_example_model('asia')
# print(asia_model)

# Visualize the network
# viz = asia_model.to_graphviz()
# viz.draw('asia.png', prog='neato')
# Image('asia.png')

# df = bn.import_example(data='titanic')
# print(df.head(5))

# model = bn.import_DAG('titanic')
# print(model)

# # df = bn.sampling(model, n=10)
# # print(df.head(5))

# bn.plot(model)


def write_excel(res_list_CAG_wald
                , res_list_CAG_anc
                , res_list_CAG_val
                , res_list_L
                , res_list_K_wald
                , res_list_K_anc
                , res_list_K_val
                , res_list_rcd_wald
                , res_list_rcd_val
                , res_list_capa_0_wald
                , res_list_capa_0_val
                , res_list_capa_1_wald
                , res_list_capa_1_val
                , res_list_capa_2_wald
                , res_list_capa_2_val
                , name):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet('CAG_wald',cell_overwrite_ok=True)
    sheet2 = f.add_sheet('CAG_anc',cell_overwrite_ok=True)
    sheet3 = f.add_sheet('CAG_val',cell_overwrite_ok=True)
    
    sheet4 = f.add_sheet('K_wald',cell_overwrite_ok=True)
    sheet5 = f.add_sheet('K_anc',cell_overwrite_ok=True)
    sheet6 = f.add_sheet('K_val',cell_overwrite_ok=True)

    sheet7 = f.add_sheet('LiNGAM',cell_overwrite_ok=True)
    sheet8 = f.add_sheet('RCD_wald',cell_overwrite_ok=True)
    sheet9 = f.add_sheet('RCD_val',cell_overwrite_ok=True)
    
    sheet10 = f.add_sheet('CAPA_0_wald',cell_overwrite_ok=True)
    sheet11 = f.add_sheet('CAPA_0_val',cell_overwrite_ok=True)
    sheet12 = f.add_sheet('CAPA_1_wald',cell_overwrite_ok=True)
    
    sheet13 = f.add_sheet('CAPA_1_val',cell_overwrite_ok=True)
    sheet14 = f.add_sheet('CAPA_2_wald',cell_overwrite_ok=True)
    sheet15 = f.add_sheet('CAPA_2_val',cell_overwrite_ok=True)
    
    for i in range(len(res_list_CAG_wald)):
        for j in range(len(res_list_CAG_wald[0])):
            sheet1.write(i,j, res_list_CAG_wald[i][j])

    for i in range(len(res_list_CAG_anc)):
        for j in range(len(res_list_CAG_anc[0])):
            sheet2.write(i,j, res_list_CAG_anc[i][j])

    for i in range(len(res_list_CAG_val)):
        for j in range(len(res_list_CAG_val[0])):
            sheet3.write(i,j, res_list_CAG_val[i][j])

    for i in range(len(res_list_K_wald)):
        for j in range(len(res_list_K_wald[0])):
            sheet4.write(i,j, res_list_K_wald[i][j])
    
    for i in range(len(res_list_K_anc)):
        for j in range(len(res_list_K_anc[0])):
            sheet5.write(i,j, res_list_K_anc[i][j])

    for i in range(len(res_list_K_val)):
        for j in range(len(res_list_K_val[0])):
            sheet6.write(i,j, res_list_K_val[i][j])
    
    for i in range(len(res_list_L)):
        for j in range(len(res_list_L[0])):
            sheet7.write(i,j, res_list_L[i][j])
    
    for i in range(len(res_list_rcd_wald)):
        for j in range(len(res_list_rcd_wald[0])):
            sheet8.write(i,j, res_list_rcd_wald[i][j])
    
    for i in range(len(res_list_rcd_val)):
        for j in range(len(res_list_rcd_val[0])):
            sheet9.write(i,j, res_list_rcd_val[i][j])

    for i in range(len(res_list_capa_0_wald)):
        for j in range(len(res_list_capa_0_wald[0])):
            sheet10.write(i,j, res_list_capa_0_wald[i][j])

    for i in range(len(res_list_capa_0_val)):
        for j in range(len(res_list_capa_0_val[0])):
            sheet11.write(i,j, res_list_capa_0_val[i][j])

    for i in range(len(res_list_capa_1_wald)):
        for j in range(len(res_list_capa_1_wald[0])):
            sheet12.write(i,j, res_list_capa_1_wald[i][j])

    for i in range(len(res_list_capa_1_val)):
        for j in range(len(res_list_capa_1_val[0])):
            sheet13.write(i,j, res_list_capa_1_val[i][j])

    for i in range(len(res_list_capa_2_wald)):
        for j in range(len(res_list_capa_2_wald[0])):
            sheet14.write(i,j, res_list_capa_2_wald[i][j])
    
    for i in range(len(res_list_capa_2_val)):
        for j in range(len(res_list_capa_2_val[0])):
            sheet15.write(i,j, res_list_capa_2_val[i][j])

    f.save(name + '.xls')


if __name__ == '__main__':
    
    current_time = time.localtime()
    formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    # alpha_l_list = [0.01, 0.05, 0.1, 0.5]
    # alpha_i_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    # alpha_l_list = [0.01, 0.05, 0.1, 0.5]
    # alpha_i_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    alpha_l_list = [0.01]
    alpha_i_list = [0.1]
    alpha_p = [1]

    samples = [15, 25, 50, 100]

    df_fmri = pd.read_excel("fMRI.xlsx", sheet_name="Sheet1", header=None)

    data_array_fmri = df_fmri.to_numpy()
    # print(data_array)
    data_array_fmri = np.where(df_fmri.to_numpy() == -1, 0, df_fmri.to_numpy())
    data_array_fmri = np.where(data_array_fmri != 0, 1.0, 0.0)

    print(data_array_fmri.T)

    # draw.draw(data_array_fmri.T)

    df_fmri_data = pd.read_excel("fMRI.xlsx", sheet_name="Sheet2", header=None)
    data_array_fmri_data = df_fmri_data.to_numpy()
    # indices = np.random.choice(data_array_fmri_data.shape[0], 100, replace=False)
    # selected_data_array_fmri_data = data_array_fmri_data[indices, : ]
    
    print(data_array_fmri_data.shape)
    # print(selected_data_array_fmri_data.shape)

    
    df_sachs_data = pd.read_excel("sachs.xls", sheet_name="Sheet2")
    data_array_sachs_data = df_sachs_data.to_numpy()
    # indices = np.random.choice(data_array_sachs_data.shape[0], 100, replace=False)
    # selected_data_array_sachs_data = data_array_sachs_data[indices, : ]
    print(data_array_sachs_data)
    # data_array_sachs_data_Z = (data_array_sachs_data - np.mean(data_array_sachs_data, axis=0)) / np.std(data_array_sachs_data, axis=0))
    # print(selected_data_array_sachs_data.shape)
    
    # input()
    # model = bn.import_DAG('sachs')
    # row_order = ["PKC","PKA", "Raf", "Jnk", "P38", "Mek", "Erk", "Akt", "Plcg", "PIP3", "PIP2"]
    # col_order = ["PKC","PKA", "Raf", "Jnk", "P38", "Mek", "Erk", "Akt", "Plcg", "PIP3", "PIP2"]
    # # print(model["adjmat"])
    # df_sachs = model["adjmat"]
    # df_sachs = df_sachs.reindex(index=row_order, columns=col_order)
    # print(df_sachs)
    # data_array_sachs = np.where(df_sachs.to_numpy(), 1.0, 0.0)
    # print(data_array_sachs.T)
    data_array_sachs=np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
            ]
        ).T
    # data_array_sachs
    
    # draw.draw(data_array_sachs.T)
    # print(model["adjmat"][2:,1:], type(model["adjmat"]))

    # ex.execute_KCI_once([df_fmri, ])
    count = 0
    for _ in range(100):
        count += 1
        current_time = time.localtime()
        formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        for s in samples:
            # RES_List = list()
            
            # node_num = 0
            for c in range(2):
                res_list_CAG_wald = list()
                res_list_CAG_anc = list()
                res_list_CAG_val = list()
                res_list_L = list()
                res_list_K_wald = list()
                res_list_K_anc = list()
                res_list_K_val = list()
                res_list_rcd_wald = list()
                res_list_rcd_val = list()
                res_list_capa_0_wald = list()
                res_list_capa_1_wald = list()
                res_list_capa_2_wald = list()
                res_list_capa_0_val = list()
                res_list_capa_1_val = list()
                res_list_capa_2_val = list()
                node_num = 0
                DAG = 0
                data = 0
                B = 0
                name = ""
                if c== 0:
                    indices = np.random.choice(data_array_fmri_data.shape[0], s, replace=False)
                    selected_data_array_fmri_data = data_array_fmri_data[indices, : ]
                    name = "fmri"
                    node_num = len(data_array_fmri)
                    DAG = data_array_fmri.T
                    data = (selected_data_array_fmri_data - np.mean(selected_data_array_fmri_data, axis=0))/np.std(selected_data_array_fmri_data, axis=0)
                    B = data_array_fmri.T
                else:
                    indices = np.random.choice(data_array_sachs_data.shape[0], s, replace=False)
                    selected_data_array_sachs_data = data_array_sachs_data[indices, : ]
                    name = "sachs"
                    node_num = len(data_array_sachs)
                    DAG = data_array_sachs.T
                    data = (selected_data_array_sachs_data - np.mean(selected_data_array_sachs_data, axis=0))/np.std(selected_data_array_sachs_data, axis=0)
                    # data = selected_data_array_sachs_data
                    B = data_array_sachs.T
                for alpha_l in alpha_l_list:
                    for alpha_i in alpha_i_list:
                        # node_num = len(data_array_fmri)
                        
                        res_CAG = np.array(ex.execute_KCI_once(
                            [
                                DAG
                                , data
                                , B
                                , alpha_l
                                , alpha_i
                                , alpha_i
                                , 1
                            ]
                            ))
                        print("CAG: ", res_CAG)
                        res_1 = res_CAG[0]
                        res_2 = res_CAG[1]
                        res_3 = res_CAG[2]
                        evaluation_list_1 = evaluation.evalute_performance_real(res_1, node_num, alpha_l, alpha_i)
                        evaluation_list_2 = evaluation.evalute_performance_real(res_2, node_num, alpha_l, alpha_i)
                        evaluation_list_3 = evaluation.evalute_performance_real(res_3, node_num, alpha_l, alpha_i)
                        print("CAG: ", [evaluation_list_1, evaluation_list_2, evaluation_list_3])
                        
                        res_list_CAG_wald.append(evaluation_list_1)
                        res_list_CAG_anc.append(evaluation_list_2)
                        res_list_CAG_val.append(evaluation_list_3)

                        res_L = ex.execute_L_once(
                            [
                                DAG
                                , data
                                , B
                            
                            ]
                        )
                        print("L: ", res_L)
                        res_list = np.array(res_L)

                        evaluation_list = evaluation.evalute_L_performance_real(res_list, 0, 0)
                        print("L: ", [evaluation_list])

                        res_list_L.append(evaluation_list)

                        res_rcd = np.array(ex.execute_RCD_once(
                            [
                                DAG
                                , data
                                , B
                                , alpha_l
                                , alpha_i
                            ]
                        ))
                        print("RCD: ", res_rcd)
                        res_1 = res_rcd[0]
                        res_2 = res_rcd[1]
                        # res_3 = res_list[:, 2, : ]

                        evaluation_list_1 = evaluation.evalute_performance_real(res_1, node_num, alpha_l, alpha_i)
                        evaluation_list_2 = evaluation.evalute_performance_real(res_2, node_num, alpha_l, alpha_i)
                        print("RCD: ",
                            [
                            evaluation_list_1
                            , [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            , evaluation_list_2
                        ])

                        res_list_rcd_wald.append(evaluation_list_1)
                        res_list_rcd_val.append(evaluation_list_2)
                        
                        res_K = np.array(ex.execute_K_once(
                            [
                                DAG
                                , data
                                , B
                            ]
                        ))
                        print("K: ", res_K)
                        res_1 = res_K[0]
                        res_2 = res_K[1]
                        res_3 = res_K[2]
                        evaluation_list_1 = evaluation.evalute_performance_real(res_1, node_num, alpha_l, alpha_i)
                        evaluation_list_2 = evaluation.evalute_performance_real(res_2, node_num, alpha_l, alpha_i)
                        evaluation_list_3 = evaluation.evalute_performance_real(res_3, node_num, alpha_l, alpha_i)
                        print("K: ", [evaluation_list_1, evaluation_list_2, evaluation_list_3])

                        res_list_K_wald.append(evaluation_list_1)
                        res_list_K_anc.append(evaluation_list_2)
                        res_list_K_val.append(evaluation_list_3)

                        res_CAPA = np.array(ex.execute_CAPA_once(
                            [
                                DAG
                                , data
                                , B
                                , 0
                                , alpha_i
                                # , alpha_i
                                # , 1
                            ]
                        ))
                        print("CAPA: ", res_CAPA)

                        res_1 = res_CAPA[0]
                        res_2 = res_CAPA[1]
                        evaluation_list_1 = evaluation.evalute_L_performance_real(res_1, 0, alpha_l)
                        evaluation_list_2 = evaluation.evalute_L_performance_real(res_2, 0, alpha_l)

                        print("CAPA: ",
                            [
                            evaluation_list_1
                            , [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            , evaluation_list_2
                            ]
                            )
                        res_list_capa_0_wald.append(evaluation_list_1)
                        # res_list_K_anc.append(evaluation_list_2)
                        res_list_capa_0_val.append(evaluation_list_2)

                        res_CAPA = np.array(ex.execute_CAPA_once(
                            [
                                DAG
                                , data
                                , B
                                , 1
                                , alpha_i
                                # , alpha_i
                                # , 1
                            ]
                        ))
                        print("CAPA: ", res_CAPA)

                        res_1 = res_CAPA[0]
                        res_2 = res_CAPA[1]
                        evaluation_list_1 = evaluation.evalute_L_performance_real(res_1, 0, alpha_l)
                        evaluation_list_2 = evaluation.evalute_L_performance_real(res_2, 0, alpha_l)

                        print("CAPA: ",
                            [
                            evaluation_list_1
                            , [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            , evaluation_list_2
                            ]
                            )
                        
                        res_list_capa_1_wald.append(evaluation_list_1)
                        res_list_capa_1_val.append(evaluation_list_2)
                        
                        res_CAPA = np.array(ex.execute_CAPA_once(
                            [
                                DAG
                                , data
                                , B
                                , 2
                                , alpha_i
                                # , alpha_i
                                # , 1
                            ]
                        ))
                        print("CAPA: ", res_CAPA)

                        res_1 = res_CAPA[0]
                        res_2 = res_CAPA[1]
                        evaluation_list_1 = evaluation.evalute_L_performance_real(res_1, 0, alpha_l)
                        evaluation_list_2 = evaluation.evalute_L_performance_real(res_2, 0, alpha_l)

                        print("CAPA: ",
                            [
                            evaluation_list_1
                            , [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            , evaluation_list_2
                            ]
                            )
                        
                        res_list_capa_2_wald.append(evaluation_list_1)
                        res_list_capa_2_val.append(evaluation_list_2)


                        write_excel(res_list_CAG_wald
                        , res_list_CAG_anc
                        , res_list_CAG_val
                        , res_list_L
                        , res_list_K_wald
                        , res_list_K_anc
                        , res_list_K_val
                        , res_list_rcd_wald
                        , res_list_rcd_val
                        , res_list_capa_0_wald
                        , res_list_capa_0_val
                        , res_list_capa_1_wald
                        , res_list_capa_1_val
                        , res_list_capa_2_wald
                        , res_list_capa_2_val
                        , name + "_" + str(s) + "_" + formatted_time + "_" + str(count)
                        )
