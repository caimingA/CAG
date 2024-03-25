import data_generator
import find_ancestor
import CAPA
import myRCD

import datetime
import numpy as np
import lingam
import lingam_local
import evaluation
import time
from multiprocessing import Pool
import copy

process_num = 18


def execute_HSIC_once(experiment_set):
    DAG_test = experiment_set[0]
    data = experiment_set[1]
    B = experiment_set[2]
    l_alpha = experiment_set[3]
    i_alpha = experiment_set[4]
    i_alpha_U = experiment_set[5]
    p_alpha = experiment_set[6]

    node_num = len(DAG_test)
    sample_size = len(experiment_set[1])
    
    adding_err_edge_1 = 0
    deleting_err_edge_1 = 0
    corr_edge_1 = 0

    adding_err_edge_2 = 0
    deleting_err_edge_2 = 0
    corr_edge_2 = 0

    adding_err_edge_3 = 0
    deleting_err_edge_3 = 0
    corr_edge_3 = 0
    
    start_time = time.time()
    ancestor_dict = dict()
    for i in range(node_num):
        ancestor_dict[i] = []

    find_ancestor.get_ancestor_loop_HSIC(data, ancestor_dict, l_alpha, i_alpha, i_alpha_U)

    AA_dict = dict()
    for i in range(node_num):
        AA_dict[i] = [i] + ancestor_dict[i]

    # ancestor_dict_list.append(AA_dict)   
    groups = find_ancestor.get_group(AA_dict, sample_size)

    M_res = find_ancestor.get_res(data, groups, B, ancestor_dict)
    end_time = time.time()
    
    M_1 = M_res[0]
    M_2 = M_res[1]
    M_3 = M_res[2]

    time_M_1 = M_res[3]
    time_M_2 = M_res[4]
    time_M_3 = M_res[5]
    
    
    for i in range(len(M_1)):
        for j in range(len(M_1)):
            if(M_1[i][j]):
                if(DAG_test[i][j]):
                    corr_edge_1 += 1
                else:
                    adding_err_edge_1 += 1
            else:
                if(DAG_test[i][j]):
                    deleting_err_edge_1 += 1

    for i in range(len(M_2)):
        for j in range(len(M_2)):
            if(M_2[i][j]):
                if(DAG_test[i][j]):
                    corr_edge_2 += 1
                else:
                    adding_err_edge_2 += 1
            else:
                if(DAG_test[i][j]):
                    deleting_err_edge_2 += 1


    for i in range(len(M_3)):
        for j in range(len(M_3)):
            if(M_3[i][j]):
                if(DAG_test[i][j]):
                    corr_edge_3 += 1
                else:
                    adding_err_edge_3 += 1
            else:
                if(DAG_test[i][j]):
                    deleting_err_edge_3 += 1
    # print(M)
    true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_test)
    count = find_ancestor.list_diff(true_ancestor_dict, ancestor_dict)
    return [
        [adding_err_edge_1, deleting_err_edge_1, corr_edge_1, count, (end_time - start_time - time_M_2 - time_M_3)],
        [adding_err_edge_2, deleting_err_edge_2, corr_edge_2, count, (end_time - start_time - time_M_1 - time_M_3)],
        [adding_err_edge_3, deleting_err_edge_3, corr_edge_3, count, (end_time - start_time - time_M_1 - time_M_2)]
        ]


def execute_HSIC(DAG_list, data_list, B_list, l_alpha, i_alpha, i_alpha_U, p_alpha=1):
    dag_size = len(DAG_list)

    node_num = len(DAG_list[0])
    
    experiment_list = list()
    for t in range(dag_size):
        experiment_list.append([DAG_list[t], data_list[t], B_list[t], l_alpha, i_alpha, i_alpha_U, p_alpha])
    # print(experiment_list)
    with Pool(process_num) as p:
        res_list = p.map(execute_HSIC_once, experiment_list)
    
    res_list = np.array(res_list)

    res_1 = res_list[:, 0, : ]
    res_2 = res_list[:, 1, : ]
    res_3 = res_list[:, 2, : ]

    # print("-----------D-----------")
    # print(res_1) 

    evaluation_list_1 = evaluation.evalute_performance(res_1, node_num, l_alpha, i_alpha)
    evaluation_list_2 = evaluation.evalute_performance(res_2, node_num, l_alpha, i_alpha)
    evaluation_list_3 = evaluation.evalute_performance(res_3, node_num, l_alpha, i_alpha)

    return [evaluation_list_1, evaluation_list_2, evaluation_list_3]


def execute_KCI_once(experiment_set):
    DAG_test = experiment_set[0]
    data = experiment_set[1]
    B = experiment_set[2]
    l_alpha = experiment_set[3]
    i_alpha = experiment_set[4]
    i_alpha_U = experiment_set[5]
    p_alpha = experiment_set[6]

    node_num = len(DAG_test)
    sample_size = len(experiment_set[1])
    
    adding_err_edge_1 = 0
    deleting_err_edge_1 = 0
    corr_edge_1 = 0
    inverse_err_edge_1 = 0

    adding_err_edge_2 = 0
    deleting_err_edge_2 = 0
    corr_edge_2 = 0
    inverse_err_edge_2 = 0

    adding_err_edge_3 = 0
    deleting_err_edge_3 = 0
    corr_edge_3 = 0
    inverse_err_edge_3 = 0
    
    no_estimated_ancestor_flag = 1

    start_time = time.time()
    ancestor_dict = dict()
    for i in range(node_num):
        ancestor_dict[i] = []

    find_ancestor.get_ancestor_loop_KCI(data, ancestor_dict, l_alpha, i_alpha, i_alpha_U)

    AA_dict = dict()
    for i in range(node_num):
        AA_dict[i] = [i] + ancestor_dict[i]
        if len(ancestor_dict[i]) != 0:
            no_estimated_ancestor_flag = 0

    # ancestor_dict_list.append(AA_dict)   
    groups = find_ancestor.get_group(AA_dict, sample_size)

    M_res = find_ancestor.get_res(data, groups, B, ancestor_dict)
    end_time = time.time()
    
    M_1 = M_res[0]
    M_2 = M_res[1]
    M_3 = M_res[2]

    time_M_1 = M_res[3]
    time_M_2 = M_res[4]
    time_M_3 = M_res[5]
        
    for i in range(1, node_num):
        for j in range(0, i):
            if(DAG_test[i][j]):
                if(M_1[i][j]): # 正确的边
                    corr_edge_1 += 1
                else: # 没有的边或者反的边
                    if(M_1[j][i]):
                        inverse_err_edge_1 += 1
                    else:
                        deleting_err_edge_1 += 1
                
                if(M_2[i][j]): # 正确的边
                    corr_edge_2 += 1
                else: # 没有的边或者反的边
                    if(M_2[j][i]):
                        inverse_err_edge_2 += 1
                    else:
                        deleting_err_edge_2 += 1
                
                if(M_3[i][j]): # 正确的边
                    corr_edge_3 += 1
                else: # 没有的边或者反的边
                    if(M_3[j][i]):
                        inverse_err_edge_3 += 1
                    else:
                        deleting_err_edge_3 += 1
            else:
                if(M_1[i][j]): # 多余的边
                    adding_err_edge_1 += 1
                else: # 正确的或者反的边
                    if(M_1[j][i]):
                        adding_err_edge_1 += 1
                    else:
                        # corr_edge_1 += 1
                        pass
                
                if(M_2[i][j]): # 多余的边
                    adding_err_edge_2 += 1
                else: # 正确的或者反的边
                    if(M_2[j][i]):
                        adding_err_edge_2 += 1
                    else:
                        # corr_edge_2 += 1
                        pass

                if(M_3[i][j]): # 多余的边
                    adding_err_edge_3 += 1
                else: # 正确的或者反的边
                    if(M_3[j][i]):
                        adding_err_edge_3 += 1
                    else:
                        # corr_edge_3 += 1
                        pass

    # print(M)
    true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_test)
    count, true_count_sum = find_ancestor.list_diff(true_ancestor_dict, ancestor_dict)

    after_ancestor_dict_1 = find_ancestor.get_true_ancestor(M_1)
    count_1, _ = find_ancestor.list_diff(true_ancestor_dict, after_ancestor_dict_1)
    after_ancestor_dict_2 = find_ancestor.get_true_ancestor(M_2)
    count_2, _ = find_ancestor.list_diff(true_ancestor_dict, after_ancestor_dict_2)
    after_ancestor_dict_3 = find_ancestor.get_true_ancestor(M_3)
    count_3, _ = find_ancestor.list_diff(true_ancestor_dict, after_ancestor_dict_3)
    return [
        [adding_err_edge_1, deleting_err_edge_1, inverse_err_edge_1, corr_edge_1, count, (end_time - start_time - time_M_2 - time_M_3), count_1, true_count_sum, no_estimated_ancestor_flag],
        [adding_err_edge_2, deleting_err_edge_2, inverse_err_edge_2, corr_edge_2, count, (end_time - start_time - time_M_1 - time_M_3), count_2, true_count_sum, no_estimated_ancestor_flag],
        [adding_err_edge_3, deleting_err_edge_3, inverse_err_edge_3, corr_edge_3, count, (end_time - start_time - time_M_1 - time_M_2), count_3, true_count_sum, no_estimated_ancestor_flag]
        ]


# def execute_KCI(DAG_list, data_list, B_list, l_alpha, i_alpha, i_alpha_U, p_alpha=1):
#     return [
#         [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]
#         ]

def execute_KCI(DAG_list, data_list, B_list, l_alpha, i_alpha, i_alpha_U, p_alpha=1):
    dag_size = len(DAG_list)

    node_num = len(DAG_list[0])
    
    experiment_list = list()
    for t in range(dag_size):
        experiment_list.append([DAG_list[t], data_list[t], B_list[t], l_alpha, i_alpha, i_alpha_U, p_alpha])
    # print(experiment_list)
    with Pool(process_num) as p:
        res_list = p.map(execute_KCI_once, experiment_list)
    
    res_list = np.array(res_list)

    res_1 = res_list[:, 0, : ]
    res_2 = res_list[:, 1, : ]
    res_3 = res_list[:, 2, : ]

    # print("-----------D-----------")
    # print(res_1) 

    evaluation_list_1 = evaluation.evalute_performance(res_1, node_num, l_alpha, i_alpha)
    evaluation_list_2 = evaluation.evalute_performance(res_2, node_num, l_alpha, i_alpha)
    evaluation_list_3 = evaluation.evalute_performance(res_3, node_num, l_alpha, i_alpha)

    
    
    return [evaluation_list_1, evaluation_list_2, evaluation_list_3]


def execute_K_once(experiment_set):
    DAG_test = experiment_set[0]
    data = experiment_set[1]
    B = experiment_set[2]

    node_num = len(DAG_test)
    sample_size = 100000
    
    adding_err_edge_1 = 0
    deleting_err_edge_1 = 0
    corr_edge_1 = 0
    inverse_err_edge_1 = 0

    adding_err_edge_2 = 0
    deleting_err_edge_2 = 0
    corr_edge_2 = 0
    inverse_err_edge_2 = 0

    adding_err_edge_3 = 0
    deleting_err_edge_3 = 0
    corr_edge_3 = 0
    inverse_err_edge_3 = 0
    
    start_time = time.time()
    
    # ancestor_dict = dict()

    true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_test)
    AA_dict = dict()
    for i in range(node_num):
        AA_dict[i] = [i] + true_ancestor_dict[i]
        
    groups = find_ancestor.get_group(AA_dict, sample_size)
        
        # print(graphs)
    M_res = find_ancestor.get_res(data, groups, B, true_ancestor_dict)
    end_time = time.time()
    
    M_1 = M_res[0]
    M_2 = M_res[1]
    M_3 = M_res[2]

    time_M_1 = M_res[3]
    time_M_2 = M_res[4]
    time_M_3 = M_res[5]
    
    
    for i in range(1, node_num):
        for j in range(0, i):
            if(DAG_test[i][j]):
                if(M_1[i][j]): # 正确的边
                    corr_edge_1 += 1
                else: # 没有的边或者反的边
                    if(M_1[j][i]):
                        inverse_err_edge_1 += 1
                    else:
                        deleting_err_edge_1 += 1
                
                if(M_2[i][j]): # 正确的边
                    corr_edge_2 += 1
                else: # 没有的边或者反的边
                    if(M_2[j][i]):
                        inverse_err_edge_2 += 1
                    else:
                        deleting_err_edge_2 += 1
                
                if(M_3[i][j]): # 正确的边
                    corr_edge_3 += 1
                else: # 没有的边或者反的边
                    if(M_3[j][i]):
                        inverse_err_edge_3 += 1
                    else:
                        deleting_err_edge_3 += 1

            else:
                if(M_1[i][j]): # 多余的边
                    adding_err_edge_1 += 1
                else: # 正确的或者反的边
                    if(M_1[j][i]):
                        adding_err_edge_1 += 1
                    else:
                        # corr_edge_1 += 1
                        pass

                if(M_2[i][j]): # 多余的边
                    adding_err_edge_2 += 1
                else: # 正确的或者反的边
                    if(M_2[j][i]):
                        adding_err_edge_2 += 1
                    else:
                        # corr_edge_2 += 1
                        pass

                if(M_3[i][j]): # 多余的边
                    adding_err_edge_3 += 1
                else: # 正确的或者反的边
                    if(M_3[j][i]):
                        adding_err_edge_3 += 1
                    else:
                        # corr_edge_3 += 1
                        pass

    # print(M)
    true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_test)
    count, true_count_sum = find_ancestor.list_diff(true_ancestor_dict, true_ancestor_dict)
    return [
        [adding_err_edge_1, deleting_err_edge_1, inverse_err_edge_1, corr_edge_1, count, (end_time - start_time - time_M_2 - time_M_3), count, true_count_sum, 0],
        [adding_err_edge_2, deleting_err_edge_2, inverse_err_edge_2, corr_edge_2, count, (end_time - start_time - time_M_1 - time_M_3), count, true_count_sum, 0],
        [adding_err_edge_3, deleting_err_edge_3, inverse_err_edge_3, corr_edge_3, count, (end_time - start_time - time_M_1 - time_M_2), count, true_count_sum, 0]
        ]

def execute_K(DAG_list, data_list, B_list):
    dag_size = len(DAG_list)

    node_num = len(DAG_list[0])
    
    experiment_list = list()

    for t in range(dag_size):
        experiment_list.append([DAG_list[t], data_list[t], B_list[t]])
    with Pool(process_num) as p:
        res_list = p.map(execute_K_once, experiment_list)

    res_list = np.array(res_list)

    res_1 = res_list[:, 0, : ]
    res_2 = res_list[:, 1, : ]
    res_3 = res_list[:, 2, : ]

    # print("-----------K-----------")
    # print(res_1)

    # time_M_1 = M_res[3]
    # time_M_2 = M_res[4]
    # time_M_3 = M_res[5]

    evaluation_list_1 = evaluation.evalute_performance(res_1, node_num, 0, 0)
    evaluation_list_2 = evaluation.evalute_performance(res_2, node_num, 0, 0)
    evaluation_list_3 = evaluation.evalute_performance(res_3, node_num, 0, 0)

    return [evaluation_list_1, evaluation_list_2, evaluation_list_3]


# def execute_K(DAG_list, data_list, B_list):
#     return [
#         [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]
#         ]


def execute_L_once(experiment_set):
    DAG_test = experiment_set[0]
    data = experiment_set[1]
    B = experiment_set[2]

    node_num = len(DAG_test)
    sample_size = len(experiment_set[1])

    # print(sample_size)

    adding_err_edge = 0
    deleting_err_edge = 0
    corr_edge = 0
    inverse_err_edge = 0
    
    start_time = time.time()

    model_L = lingam.DirectLiNGAM()
    model_L.fit(data)
    m_L = model_L.adjacency_matrix_

    end_time = time.time()

    for i in range(1, node_num):
        for j in range(0, i):
            if(DAG_test[i][j]):
                if(m_L[i][j]): # 正确的边
                    corr_edge += 1
                else: # 没有的边或者反的边
                    if(m_L[j][i]):
                        inverse_err_edge += 1
                    else:
                        deleting_err_edge += 1
            else:
                if(m_L[i][j]): # 多余的边
                    adding_err_edge += 1
                else: # 正确的或者反的边
                    if(m_L[j][i]):
                        adding_err_edge += 1
                    else:
                        # corr_edge += 1
                        pass
    
    return [adding_err_edge, deleting_err_edge, inverse_err_edge, corr_edge, (end_time - start_time)]


def execute_L(DAG_list, data_list, B_list):
    dag_size = len(DAG_list)

    node_num = len(DAG_list[0])
    
    experiment_list = list()

    for t in range(dag_size):
        experiment_list.append([DAG_list[t], data_list[t], B_list[t]])
    with Pool(process_num) as p:
        res_list = p.map(execute_L_once, experiment_list)

    res_list = np.array(res_list)

    evaluation_list = evaluation.evalute_L_performance(res_list, 0, 0)

    return evaluation_list


# def execute_L(DAG_list, data_list, B_list):
#     return [0, 0, 0, 0, 0, 0, 0]


# def execute_RCD_once(experiment_set):
#     DAG_test = experiment_set[0]
#     data = experiment_set[1]
#     B = experiment_set[2]
#     l_alpha = experiment_set[3]
#     i_alpha = experiment_set[4]

#     node_num = len(DAG_test)
#     sample_size = len(experiment_set[1])

#     adding_err_edge_1 = 0
#     deleting_err_edge_1 = 0
#     corr_edge_1 = 0
#     inverse_err_edge_1 = 0

#     adding_err_edge_2 = 0
#     deleting_err_edge_2 = 0
#     corr_edge_2 = 0
#     inverse_err_edge_2 = 0
    
#     start_time = time.time()

#     model_RCD = lingam.RCD(
#         max_explanatory_num=2
#        , cor_alpha=l_alpha
#        , ind_alpha=i_alpha
#        , shapiro_alpha=1
#        , MLHSICR=True
#     )

#     model_RCD.fit(data)
#     m_L = model_RCD.adjacency_matrix_

#     for i in range(node_num):
#         for j in range(node_num):
#             if np.isnan(m_L[i][j]):
#                 m_L[i][j] = 0

    
#     ancestor_dict = model_RCD._ancestors_list
#     # count = 0
#     m_L_1 = copy.deepcopy(m_L)
#     m_L_2 = copy.deepcopy(m_L)

#     temp_time = time.time()

#     temp_loop = find_ancestor.find_loop(m_L_1)
#     while(len(temp_loop)):
#         # print(True)
#         m_L_1 = find_ancestor.wald_eliminate_loop(m_L_1, temp_loop, data)
#         temp_loop = find_ancestor.find_loop(m_L_1)
    
#     mid_time = time.time()
    
#     temp_loop = find_ancestor.find_loop(m_L_2)
#     while(len(temp_loop)):
#         m_L_2 = find_ancestor.value_eliminate_loop(m_L_2, temp_loop)
#         temp_loop = find_ancestor.find_loop(m_L_2)
    
#     end_time = time.time()

#     for i in range(1, node_num):
#         for j in range(0, i):
#             if(DAG_test[i][j]):
#                 if(m_L_1[i][j]): # 正确的边
#                     corr_edge_1 += 1
#                 else: # 没有的边或者反的边
#                     if(m_L_1[j][i]):
#                         inverse_err_edge_1 += 1
#                     else:
#                         deleting_err_edge_1 += 1
#             else:
#                 if(m_L_1[i][j]): # 多余的边
#                     adding_err_edge_1 += 1
#                 else: # 正确的或者反的边
#                     if(m_L_1[j][i]):
#                         adding_err_edge_1 += 1
#                     else:
#                         # corr_edge += 1
#                         pass
    
#     for i in range(1, node_num):
#         for j in range(0, i):
#             if(DAG_test[i][j]):
#                 if(m_L_2[i][j]): # 正确的边
#                     corr_edge_2 += 1
#                 else: # 没有的边或者反的边
#                     if(m_L_2[j][i]):
#                         inverse_err_edge_2 += 1
#                     else:
#                         deleting_err_edge_2 += 1
#             else:
#                 if(m_L_2[i][j]): # 多余的边
#                     adding_err_edge_2 += 1
#                 else: # 正确的或者反的边
#                     if(m_L_2[j][i]):
#                         adding_err_edge_2 += 1
#                     else:
#                         # corr_edge += 1
#                         pass

#     true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_test)
#     count = find_ancestor.list_diff(ancestor_dict, true_ancestor_dict)

#     after_ancestor_dict_1 = find_ancestor.get_true_ancestor(m_L_1)
#     count_1 = find_ancestor.list_diff(after_ancestor_dict_1, true_ancestor_dict)

#     after_ancestor_dict_2 = find_ancestor.get_true_ancestor(m_L_2)
#     count_2 = find_ancestor.list_diff(after_ancestor_dict_2, true_ancestor_dict)
    
#     return [
#         [adding_err_edge_1, deleting_err_edge_1, inverse_err_edge_1, corr_edge_1, count, (mid_time - start_time), count_1],
#         [adding_err_edge_2, deleting_err_edge_2, inverse_err_edge_2, corr_edge_2, count, (end_time - start_time) - (mid_time - temp_time), count_2]
#         ]


def execute_RCD_once(experiment_set):
    DAG_test = experiment_set[0]
    data = experiment_set[1]
    B = experiment_set[2]
    l_alpha = experiment_set[3]
    i_alpha = experiment_set[4]

    node_num = len(DAG_test)
    sample_size = len(experiment_set[1])
    
    adding_err_edge_1 = 0
    deleting_err_edge_1 = 0
    corr_edge_1 = 0
    inverse_err_edge_1 = 0

    # adding_err_edge_2 = 0
    # deleting_err_edge_2 = 0
    # corr_edge_2 = 0
    # inverse_err_edge_2 = 0

    adding_err_edge_3 = 0
    deleting_err_edge_3 = 0
    corr_edge_3 = 0
    inverse_err_edge_3 = 0

    no_estimated_ancestor_flag = 1
    
    start_time = time.time()
    ancestor_dict = dict()
    for i in range(node_num):
        ancestor_dict[i] = []

    find_ancestor.get_ancestor_loop_KCI(data, ancestor_dict, l_alpha, i_alpha, i_alpha)

    AA_dict = dict()
    for i in range(node_num):
        AA_dict[i] = [i] + ancestor_dict[i]
        if len(ancestor_dict[i]) != 0:
            no_estimated_ancestor_flag = 0

    # ancestor_dict_list.append(AA_dict)   
    groups = find_ancestor.get_group(AA_dict, sample_size)

    ancestor_dict = myRCD.change_list_to_set(ancestor_dict)

    M_res_rcd = myRCD.get_res(data, B, ancestor_dict, i_alpha)
    end_time = time.time()
    
    M_1 = M_res_rcd[0]
    M_2 = M_res_rcd[1]
    M_3 = M_res_rcd[2]

    time_M_1 = M_res_rcd[3]
    time_M_2 = M_res_rcd[4]
    time_M_3 = M_res_rcd[5]
        
    for i in range(1, node_num):
        for j in range(0, i):
            if(DAG_test[i][j]):
                if(M_1[i][j]): # 正确的边
                    corr_edge_1 += 1
                else: # 没有的边或者反的边
                    if(M_1[j][i]):
                        inverse_err_edge_1 += 1
                    else:
                        deleting_err_edge_1 += 1
                
                # if(M_2[i][j]): # 正确的边
                #     corr_edge_2 += 1
                # else: # 没有的边或者反的边
                #     if(M_2[j][i]):
                #         inverse_err_edge_2 += 1
                #     else:
                #         deleting_err_edge_2 += 1
                
                if(M_3[i][j]): # 正确的边
                    corr_edge_3 += 1
                else: # 没有的边或者反的边
                    if(M_3[j][i]):
                        inverse_err_edge_3 += 1
                    else:
                        deleting_err_edge_3 += 1
            else:
                if(M_1[i][j]): # 多余的边
                    adding_err_edge_1 += 1
                else: # 正确的或者反的边
                    if(M_1[j][i]):
                        adding_err_edge_1 += 1
                    else:
                        # corr_edge_1 += 1
                        pass
                
                # if(M_2[i][j]): # 多余的边
                #     adding_err_edge_2 += 1
                # else: # 正确的或者反的边
                #     if(M_2[j][i]):
                #         adding_err_edge_2 += 1
                #     else:
                #         # corr_edge_2 += 1
                #         pass

                if(M_3[i][j]): # 多余的边
                    adding_err_edge_3 += 1
                else: # 正确的或者反的边
                    if(M_3[j][i]):
                        adding_err_edge_3 += 1
                    else:
                        # corr_edge_3 += 1
                        pass

    # print(M)
    true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_test)
    count, true_count_sum = find_ancestor.list_diff(true_ancestor_dict, ancestor_dict)

    after_ancestor_dict_1 = find_ancestor.get_true_ancestor(M_1)
    count_1, _ = find_ancestor.list_diff(true_ancestor_dict, after_ancestor_dict_1)
    # after_ancestor_dict_2 = find_ancestor.get_true_ancestor(M_2)
    # count_2, _ = find_ancestor.list_diff(true_ancestor_dict, after_ancestor_dict_2)
    after_ancestor_dict_3 = find_ancestor.get_true_ancestor(M_3)
    count_3, _ = find_ancestor.list_diff(true_ancestor_dict, after_ancestor_dict_3)
    return [
        [adding_err_edge_1, deleting_err_edge_1, inverse_err_edge_1, corr_edge_1, count, (end_time - start_time - time_M_2 - time_M_3), count_1, true_count_sum, no_estimated_ancestor_flag],
        # [adding_err_edge_2, deleting_err_edge_2, inverse_err_edge_2, corr_edge_2, count, (end_time - start_time - time_M_1 - time_M_3), count_2],
        [adding_err_edge_3, deleting_err_edge_3, inverse_err_edge_3, corr_edge_3, count, (end_time - start_time - time_M_1 - time_M_2), count_3, true_count_sum, no_estimated_ancestor_flag]
        ]


def execute_RCD(DAG_list, data_list, B_list, l_alpha, i_alpha):
    dag_size = len(DAG_list)

    node_num = len(DAG_list[0])
    
    experiment_list = list()

    for t in range(dag_size):
        experiment_list.append([DAG_list[t], data_list[t], B_list[t], l_alpha, i_alpha])
    with Pool(process_num) as p:
        res_list = p.map(execute_RCD_once, experiment_list)

    res_list = np.array(res_list)

    # evaluation_list = evaluation.evalute_performance(res_list, node_num)
    res_1 = res_list[:, 0, : ]
    res_2 = res_list[:, 1, : ]
    # res_3 = res_list[:, 2, : ]

    # print("-----------K-----------")
    # print(res_1)

    # time_M_1 = M_res[3]
    # time_M_2 = M_res[4]
    # time_M_3 = M_res[5]

    evaluation_list_1 = evaluation.evalute_performance(res_1, node_num, l_alpha, i_alpha)
    evaluation_list_2 = evaluation.evalute_performance(res_2, node_num, l_alpha, i_alpha)
    # evaluation_list_3 = evaluation.evalute_performance(res_3, node_num)

    return [
        evaluation_list_1
        , [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        , evaluation_list_2
        ]


def execute_CAPA_once(experiment_set):
    DAG_test = experiment_set[0]
    data = experiment_set[1]
    B = experiment_set[2]

    sigma = experiment_set[3]
    pVal = experiment_set[4]
    # l_alpha = experiment_set[3]
    # i_alpha = experiment_set[4]
    # i_alpha_U = experiment_set[5]
    # p_alpha = experiment_set[6]

    node_num = len(DAG_test)
    sample_size = len(experiment_set[1])
    
    adding_err_edge_1 = 0
    deleting_err_edge_1 = 0
    corr_edge_1 = 0
    inverse_err_edge_1 = 0

    adding_err_edge_2 = 0
    deleting_err_edge_2 = 0
    corr_edge_2 = 0
    inverse_err_edge_2 = 0

    # adding_err_edge_3 = 0
    # deleting_err_edge_3 = 0
    # corr_edge_3 = 0
    # inverse_err_edge_3 = 0
    
    start_time = time.time()

    # ancestor_dict_list.append(AA_dict)   
    groups = CAPA.CAPA(data, sigma, pVal)

    M_res = find_ancestor.get_res_no_ancestor(data, groups, B)
    end_time = time.time()
    
    M_1 = M_res[0]
    M_2 = M_res[1]
    # M_3 = M_res[2]

    time_M_1 = M_res[2]
    time_M_2 = M_res[3]
    # time_M_3 = M_res[5]
    
    
    for i in range(1, node_num):
        for j in range(0, i):
            if(DAG_test[i][j]):
                if(M_1[i][j]): # 正确的边
                    corr_edge_1 += 1
                else: # 没有的边或者反的边
                    if(M_1[j][i]):
                        inverse_err_edge_1 += 1
                    else:
                        deleting_err_edge_1 += 1
                
                if(M_2[i][j]): # 正确的边
                    corr_edge_2 += 1
                else: # 没有的边或者反的边
                    if(M_2[j][i]):
                        inverse_err_edge_2 += 1
                    else:
                        deleting_err_edge_2 += 1
                
                # if(M_3[i][j]): # 正确的边
                #     corr_edge_3 += 1
                # else: # 没有的边或者反的边
                #     if(M_3[j][i]):
                #         inverse_err_edge_3 += 1
                #     else:
                #         deleting_err_edge_3 += 1
            else:
                if(M_1[i][j]): # 多余的边
                    adding_err_edge_1 += 1
                else: # 正确的或者反的边
                    if(M_1[j][i]):
                        inverse_err_edge_1 += 1
                    else:
                        # corr_edge_1 += 1
                        pass
                
                if(M_2[i][j]): # 多余的边
                    adding_err_edge_2 += 1
                else: # 正确的或者反的边
                    if(M_2[j][i]):
                        inverse_err_edge_2 += 1
                    else:
                        # corr_edge_2 += 1
                        pass

                # if(M_3[i][j]): # 多余的边
                #     adding_err_edge_3 += 1
                # else: # 正确的或者反的边
                #     if(M_3[j][i]):
                #         inverse_err_edge_3 += 1
                #     else:
                #         # corr_edge_3 += 1
                #         pass

    # print(M)
    # true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_test)
    # count = find_ancestor.list_diff(true_ancestor_dict, ancestor_dict)
    return [
        [adding_err_edge_1, deleting_err_edge_1, inverse_err_edge_1, corr_edge_1, (end_time - start_time - time_M_2)],
        [adding_err_edge_2, deleting_err_edge_2, inverse_err_edge_2, corr_edge_2, (end_time - start_time - time_M_1)],
        # [adding_err_edge_3, deleting_err_edge_3, inverse_err_edge_3, corr_edge_3, (end_time - start_time - time_M_1 - time_M_2)]
        ]

def execute_CAPA(DAG_list, data_list, B_list, sigma, pVal):
    dag_size = len(DAG_list)

    node_num = len(DAG_list[0])
    
    experiment_list = list()

    for t in range(dag_size):
        experiment_list.append([DAG_list[t], data_list[t], B_list[t], sigma, pVal])
    with Pool(process_num) as p:
        res_list = p.map(execute_CAPA_once, experiment_list)

    res_list = np.array(res_list)

    res_1 = res_list[:, 0, : ]
    res_2 = res_list[:, 1, : ]
    evaluation_list_1 = evaluation.evalute_L_performance(res_1, 0, pVal)
    evaluation_list_2 = evaluation.evalute_L_performance(res_2, 0, pVal)

    return [
        evaluation_list_1
        , [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        , evaluation_list_2
        ]


def execute_test(DAG_list, data_list, B_list, l_alpha, i_alpha, i_alpha_U, p_alpha=1):
    dag_size = len(DAG_list)
    sample_size = len(data_list[0])
    
    # adding_list_pruning = list()
    # deleting_list_pruning = list()
    # corr_list_pruning = list()
    ancestor_dict_list = list()
    count_list = list()

    diff_err_list_1 = list()
    diff_err_list_2 = list()
    diff_err_list_3 = list()

    node_num = len(DAG_list[0])
    start = datetime.datetime.now()

    # print(dag_size)
    # print(sample_size)
    # print(data_list[0].shape)
    
    for t in range(dag_size):
        print(t)
        # temp_list = list()
        ancestor_dict = dict()
        for i in range(node_num):
            ancestor_dict[i] = []
        
        # print("epcho: ", t)
        adding_err_edge = 0
        deleting_err_edge = 0
        corr_edge = 0

        DAG_test = DAG_list[t]

#         count_list = list()
        true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_test)
        
        find_ancestor.get_ancestor_loop_HSIC(data_list[t], ancestor_dict, l_alpha, i_alpha, i_alpha_U, p_alpha)
#         get_ancestor_loop_3(data_list[t], ancestor_dict, l_alpha, i_alpha, i_alpha_U)
        
        count_list.append(find_ancestor.list_diff(true_ancestor_dict, ancestor_dict))
        
    
        
        AA_dict = dict()
        for i in range(node_num):
            AA_dict[i] = [i] + ancestor_dict[i]
        
        
        
        ancestor_dict_list.append(AA_dict)   
        groups = find_ancestor.get_group(AA_dict, sample_size)
        
        # print(graphs)
        # M = find_ancestor.get_res(data_list[t], groups, B_list[t])

        M_res = find_ancestor.get_res(data_list[t], groups, B_list[t], ancestor_dict)
    # end_time = time.time()
    
        M_1 = M_res[0]
        M_2 = M_res[1]
        M_3 = M_res[2]
        
        err_1 = evaluation.differ_from_ancestor(true_ancestor_dict, ancestor_dict, M_1)
        err_2 = evaluation.differ_from_ancestor(true_ancestor_dict, ancestor_dict, M_2)
        err_3 = evaluation.differ_from_ancestor(true_ancestor_dict, ancestor_dict, M_3)
        
        if len(diff_err_list_1) == 0:
            diff_err_list_1 = np.array(err_1)
        else:
            diff_err_list_1 += np.array(err_1)
        
        if len(diff_err_list_2) == 0:
            diff_err_list_2 = np.array(err_2)
        else:
            diff_err_list_2 += np.array(err_2)
        
        if len(diff_err_list_3) == 0:
            diff_err_list_3 = np.array(err_3)
        else:
            diff_err_list_3 += np.array(err_3)
    
    return [diff_err_list_1, diff_err_list_2, diff_err_list_3]