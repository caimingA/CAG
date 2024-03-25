import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def evalute_performance(res_list, node_num, l_alpha, i_alpha):
    dag_size = len(res_list)
    DAG_count = 0
    DAG_count_1 = 0
    DAG_count_2 = 0
    DAG_count_3 = 0

    adding_list = res_list[:, 0]
    deleting_list = res_list[:, 1]
    inverse_list = res_list[:, 2]
    corr_list = res_list[:, 3]
    count_list = res_list[:, 4]
    time_list = res_list[:, 5]
    after_count_list = res_list[:, 6]
    true_count_sum_list = res_list[:, 7]
    no_estimated_ancestor_flags = res_list[:, 8]

    print(
        np.sum(corr_list) 
        + np.sum(deleting_list) 
        + np.sum(adding_list) 
        + np.sum(inverse_list)
        )
    
    for i in range(dag_size):
        if (adding_list[i] + deleting_list[i] + inverse_list[i]) <= 0:
            DAG_count += 1
        if (adding_list[i] + deleting_list[i] + inverse_list[i]) <= 1:
            DAG_count_1 += 1
        if (adding_list[i] + deleting_list[i] + inverse_list[i]) <= 2:
            DAG_count_2 += 1    
        if (adding_list[i] + deleting_list[i] + inverse_list[i]) <= 3:
            DAG_count_3 += 1

    TP = np.sum(corr_list)
    TP_FP = TP + np.sum(adding_list) + np.sum(inverse_list)
    TP_FN = TP + np.sum(deleting_list) + np.sum(inverse_list)
    # TP_FN = TP + np.sum(deleting_list)

    SHD = (np.sum(adding_list) + np.sum(inverse_list) + np.sum(deleting_list)) / dag_size

    precision = TP / TP_FP if TP_FP != 0 else 0
    recall = TP / TP_FN if TP_FN != 0 else 0
    F = 2*precision*recall / (precision + recall) if (precision + recall) != 0 else 0

    ancestor_err = (np.sum(count_list) / np.sum(true_count_sum_list))
    after_ancestor_err = (np.sum(after_count_list) / np.sum(true_count_sum_list))

    execute_time = np.sum(time_list)

    no_estimated_ancestor_count = np.sum(no_estimated_ancestor_flags)
    return [l_alpha, i_alpha, precision, recall, F, SHD, DAG_count, DAG_count_1, DAG_count_2, DAG_count_3, execute_time, ancestor_err, after_ancestor_err, no_estimated_ancestor_count]


def evalute_L_performance(res_list, l_alpha, i_alpha):
    dag_size = len(res_list)
    DAG_count = 0
    DAG_count_1 = 0
    DAG_count_2 = 0
    DAG_count_3 = 0


    adding_list = res_list[:, 0]
    deleting_list = res_list[:, 1]
    inverse_list = res_list[:, 2]
    corr_list = res_list[:, 3]
    # count_list = res_list[:, 3]
    time_list = res_list[:, 4]

    for i in range(dag_size):
        if (adding_list[i] + deleting_list[i] + inverse_list[i]) <= 0:
            DAG_count += 1
        if (adding_list[i] + deleting_list[i] + inverse_list[i]) <= 1:
            DAG_count_1 += 1
        if (adding_list[i] + deleting_list[i] + inverse_list[i]) <= 2:
            DAG_count_2 += 1    
        if (adding_list[i] + deleting_list[i] + inverse_list[i]) <= 3:
            DAG_count_3 += 1

    TP = np.sum(corr_list)
    TP_FP = TP + np.sum(adding_list) + np.sum(inverse_list)
    TP_FN = TP + np.sum(deleting_list) + np.sum(inverse_list)
    # TP_FN = TP + np.sum(deleting_list)

    SHD = (np.sum(adding_list) + np.sum(inverse_list) + np.sum(deleting_list)) / dag_size
    
    precision = TP / TP_FP
    recall = TP / TP_FN
    F = 2*precision*recall / (precision + recall)

    # ancestor_err = 1 - (np.sum(count_list) / (dag_size * node_num))

    execute_time = np.sum(time_list)

    return [l_alpha, i_alpha, precision, recall, F, SHD, DAG_count, DAG_count_1, DAG_count_2, DAG_count_3, execute_time, 0]


def differ_from_ancestor(true_ancestor_dict, ancestor_dict, M):
    num = len(M)
    G = nx.DiGraph()
    for i in range(len(M)):
        G.add_node(i)
        for j in range(len(M)):
            if(M[i][j]):
                G.add_edge(j, i)

    ancestor_correct_count = 0
    LiNGAM_correct_count = 0

    ancestor_err_count = 0
    LiNGAM_err_count = 0


    plus_err_count = 0
    minus_err_count = 0

    both_correct_count = 0
    both_err_count = 0

    for i in range(1, num):
        for j in range(0, i):
            if i in true_ancestor_dict[j]: # i is the ancestor of j
                if nx.has_path(G, i, j): # 推测的图中i -> j
                    LiNGAM_correct_count += 1
                    if i in ancestor_dict[j]: # i is estimated as the ancestor of j
                        ancestor_correct_count += 1
                        both_correct_count += 1
                    else:
                        ancestor_err_count +=1
                        plus_err_count += 1
                else:
                    LiNGAM_err_count += 1
                    if i in ancestor_dict[j]: # i is estimated as the ancestor of j
                        ancestor_correct_count += 1
                        minus_err_count += 1
                    else:
                        ancestor_err_count +=1
                        both_err_count += 1
            elif j in true_ancestor_dict[i]: # j is the ancestor of i
                if nx.has_path(G, j, i): # 推测的图中j -> i
                    LiNGAM_correct_count += 1
                    if j in ancestor_dict[i]:
                        ancestor_correct_count += 1
                        both_correct_count += 1
                    else:
                        ancestor_err_count +=1
                        plus_err_count += 1
                else:
                    LiNGAM_err_count += 1
                    if j in ancestor_dict[i]:
                        ancestor_correct_count += 1
                        minus_err_count += 1
                    else:
                        ancestor_err_count += 1
                        both_err_count += 1
            else: # no ancestral relationships between i and j
                if not (nx.has_path(G, i, j) or nx.has_path(G, j, i)):
                    LiNGAM_correct_count += 1
                    if i not in ancestor_dict[j] and j not in ancestor_dict[i]:
                        ancestor_correct_count += 1
                        both_correct_count += 1
                    else:
                        ancestor_err_count += 1
                        plus_err_count += 1
                else:
                    LiNGAM_err_count += 1
                    if i not in ancestor_dict[j] and j not in ancestor_dict[i]:
                        ancestor_correct_count += 1
                        minus_err_count += 1
                    else:
                        ancestor_err_count += 1
                        both_err_count += 1
    return plus_err_count, minus_err_count, both_correct_count, both_err_count, ancestor_correct_count, ancestor_err_count, LiNGAM_correct_count, LiNGAM_err_count
            # if nx.has_path(G, i, j): # 推测的图中i -> j
            #     pass
            # elif nx.has_path(G, j, i): # 推测的图中i -> j
            #     pass
            # else: # 推测的图中i -> j
            #     pass
    
    # plus_add_err_count = 0
    # minus_add_err_count = 0
    # plus_del_err_count = 0
    # minus_del_err_count = 0

    # for i in range(1, num):
    #     for j in range(0, i):
    #         # print(i, ", ", j)
    #         # i is an ancestor of j
    #         if nx.has_path(G, i, j): # 推测的图中i -> j
    #             if i not in ancestor_dict[j]: # 推测的祖先中没这个关系
    #                 if i in ture_ancestor_dict[j]: # 结果没有遵从祖先关系但与真的祖先关系相符
    #                     plus_add_err_count += 1 
    #                 else:
    #                     minus_add_err_count += 1
    #         # j is an ancestor of i
    #         elif nx.has_path(G, j, i): # 推测的图中i -> j
    #             if j not in ancestor_dict[i]: # 推测的祖先中没这个关系
    #                 if j in ture_ancestor_dict[i]: # 结果没有遵从祖先关系但与真的祖先关系相符
    #                     plus_add_err_count += 1
    #                 else:
    #                     minus_add_err_count += 1
    #         else: # 推测的图中i 和 j 无关
    #             if i in ancestor_dict[j]: # 推测的关系中j -> i 
    #                 if i not in ture_ancestor_dict[j]: # 结果没有遵从祖先关系但与真的祖先关系相符
    #                     plus_del_err_count += 1
    #                 else:
    #                     minus_del_err_count += 1
    #             if j in ancestor_dict[i]: # 推测的关系中i -> j 
    #                 if j not in ture_ancestor_dict[i]: # 结果没有遵从祖先关系但与真的祖先关系相符
    #                     plus_del_err_count += 1
    #                 else:
    #                     minus_del_err_count += 1

    # N = num*(num - 1) / 2        
    
    # err_count = plus_del_err_count + plus_add_err_count + minus_del_err_count + minus_add_err_count
    # plus_err_count = plus_del_err_count + plus_add_err_count
    # minus_err_count = minus_del_err_count + minus_add_err_count
    # add_err_count = plus_add_err_count + minus_add_err_count
    # del_err_count = plus_del_err_count + minus_del_err_count
    # return  plus_add_err_count+plus_del_err_count, minus_add_err_count+minus_del_err_count