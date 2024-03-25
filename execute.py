import data_generator
import find_ancestor

import datetime
import numpy as np
import lingam
import lingam_local
import evaluation


def execute_HSIC(DAG_list, data_list, B_list, l_alpha, i_alpha, i_alpha_U, p_alpha=1):
    dag_size = len(DAG_list)
    sample_size = len(data_list[0])
    
    adding_list_pruning = list()
    deleting_list_pruning = list()
    corr_list_pruning = list()
    ancestor_dict_list = list()
    count_list = list()

    node_num = len(DAG_list[0])
    start = datetime.datetime.now()
    
    for t in range(dag_size):
        print(t)
        ancestor_dict = dict()
        for i in range(node_num):
            ancestor_dict[i] = []

        adding_err_edge = 0
        deleting_err_edge = 0
        corr_edge = 0

        DAG_test = DAG_list[t]

        true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_test)
        
        find_ancestor.get_ancestor_loop_HSIC(data_list[t], ancestor_dict, l_alpha, i_alpha, i_alpha_U, p_alpha)
        
        count_list.append(find_ancestor.list_diff(true_ancestor_dict, ancestor_dict))
        
        AA_dict = dict()
        for i in range(node_num):
            AA_dict[i] = [i] + ancestor_dict[i]
        
        ancestor_dict_list.append(AA_dict)   
        groups = find_ancestor.get_group(AA_dict, sample_size)

        M = find_ancestor.get_res(data_list[t], groups, B_list[t])
        
        for i in range(len(M)):
            for j in range(len(M)):
                if(M[i][j]):
                    if(DAG_test[i][j]):
                        corr_edge += 1

                    else:
                        adding_err_edge += 1
                else:
                    if(DAG_test[i][j]):
                        deleting_err_edge += 1
    
        adding_list_pruning.append(adding_err_edge)
        deleting_list_pruning.append(deleting_err_edge)
        corr_list_pruning.append(corr_edge)

    end = datetime.datetime.now()

    execute_time = (end - start).seconds
    
    DAG_count = 0
    DAG_count_3 = 0
    for i in range(len(adding_list_pruning)):
        if (adding_list_pruning[i] + deleting_list_pruning[i]) <= 0:
            DAG_count += 1
        if (adding_list_pruning[i] + deleting_list_pruning[i]) <= 3:
            DAG_count_3 += 1


    TP = np.sum(corr_list_pruning)
    TP_FP = TP + np.sum(adding_list_pruning)
    TP_FN = TP + np.sum(deleting_list_pruning)

    precision = TP / TP_FP
    recall = TP / TP_FN
    F = 2*precision*recall / (precision + recall)

    ancestor_err = 1 - (np.sum(count_list) / (dag_size * node_num))
    
    return precision, recall, F, DAG_count, DAG_count_3, execute_time, ancestor_err


def execute_HSIC_eliminate(DAG_list, data_list, B_list, l_alpha, i_alpha, i_alpha_U, p_alpha=1):
    dag_size = len(DAG_list)
    sample_size = len(data_list[0])
    
    adding_list_pruning = list()
    deleting_list_pruning = list()
    corr_list_pruning = list()
    ancestor_dict_list = list()
    count_list = list()

    diff_err_list = list()

    node_num = len(DAG_list[0])
    start = datetime.datetime.now()
    
    for t in range(dag_size):
        print(t)
        ancestor_dict = dict()
        for i in range(node_num):
            ancestor_dict[i] = []

        adding_err_edge = 0
        deleting_err_edge = 0
        corr_edge = 0

        DAG_test = DAG_list[t]

        true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_test)
        
        find_ancestor.get_ancestor_loop_HSIC(data_list[t], ancestor_dict, l_alpha, i_alpha, i_alpha_U, p_alpha)
        
        count_list.append(find_ancestor.list_diff(true_ancestor_dict, ancestor_dict))
        
    
        
        AA_dict = dict()
        for i in range(node_num):
            AA_dict[i] = [i] + ancestor_dict[i]
        
        
        
        ancestor_dict_list.append(AA_dict)   
        groups = find_ancestor.get_group(AA_dict, sample_size)
        
        M = find_ancestor.get_res(data_list[t], groups, B_list[t])
        
        M = eliminate_loop(M, ancestor_dict)
        
        for i in range(len(M)):
            for j in range(len(M)):
                if(M[i][j]):
                    if(DAG_test[i][j]):
                        corr_edge += 1

                    else:
                        adding_err_edge += 1
                else:
                    if(DAG_test[i][j]):
                        deleting_err_edge += 1
    
        adding_list_pruning.append(adding_err_edge)
        deleting_list_pruning.append(deleting_err_edge)
        corr_list_pruning.append(corr_edge)

    end = datetime.datetime.now()

    execute_time = (end - start).seconds

    DAG_count = 0
    DAG_count_3 = 0
    for i in range(len(adding_list_pruning)):
        if (adding_list_pruning[i] + deleting_list_pruning[i]) <= 0:
            DAG_count += 1
        if (adding_list_pruning[i] + deleting_list_pruning[i]) <= 3:
            DAG_count_3 += 1


    TP = np.sum(corr_list_pruning)
    TP_FP = TP + np.sum(adding_list_pruning)
    TP_FN = TP + np.sum(deleting_list_pruning)

    precision = TP / TP_FP
    recall = TP / TP_FN
    F = 2*precision*recall / (precision + recall)

    ancestor_err = 1 - (np.sum(count_list) / (dag_size * node_num))
    
    return precision, recall, F, DAG_count, DAG_count_3, execute_time, ancestor_err


def execute_K(DAG_list, data_list, B_list):
    dag_size = len(DAG_list)
    
    adding_list_pruning = list()
    deleting_list_pruning = list()
    corr_list_pruning = list()
    ancestor_dict_list = list()
    count_list = list()

    node_num = len(DAG_list[0])
    start = datetime.datetime.now()

    for t in range(dag_size):
        print(t)
        ancestor_dict = dict()
        for i in range(node_num):
            ancestor_dict[i] = []
        
        adding_err_edge = 0
        deleting_err_edge = 0
        corr_edge = 0

        DAG_test = DAG_list[t]

        true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_test)
        
        count_list.append(find_ancestor.list_diff(true_ancestor_dict, true_ancestor_dict))
                
        AA_dict = dict()
        for i in range(node_num):
            AA_dict[i] = [i] + true_ancestor_dict[i]
        
        
        
        ancestor_dict_list.append(AA_dict)   
        groups = find_ancestor.get_group(AA_dict, 10000)
        
        M = find_ancestor.get_res(data_list[t], groups, B_list[t])
        
    
        for i in range(len(M)):
            for j in range(len(M)):
                if(M[i][j]):
                    if(DAG_test[i][j]):
                        corr_edge += 1

                    else:
                        adding_err_edge += 1
                else:
                    if(DAG_test[i][j]):
                        deleting_err_edge += 1
    
        adding_list_pruning.append(adding_err_edge)
        deleting_list_pruning.append(deleting_err_edge)
        corr_list_pruning.append(corr_edge)

    end = datetime.datetime.now()

    execute_time = (end - start).seconds
    DAG_count = 0
    DAG_count_3 = 0
    for i in range(len(adding_list_pruning)):
        if (adding_list_pruning[i] + deleting_list_pruning[i]) <= 0:
            DAG_count += 1
        if (adding_list_pruning[i] + deleting_list_pruning[i]) <= 3:
            DAG_count_3 += 1


    TP = np.sum(corr_list_pruning)
    TP_FP = TP + np.sum(adding_list_pruning)
    TP_FN = TP + np.sum(deleting_list_pruning)

    precision = TP / TP_FP
    recall = TP / TP_FN
    F = 2*precision*recall / (precision + recall)

    ancestor_err = 1 - (np.sum(count_list) / (dag_size * node_num))
    
    return precision, recall, F, DAG_count, DAG_count_3, execute_time, ancestor_err


def execute_K_eliminate(DAG_list, data_list, B_list):
    dag_size = len(DAG_list)
    
    adding_list_pruning = list()
    deleting_list_pruning = list()
    corr_list_pruning = list()
    ancestor_dict_list = list()
    count_list = list()

    node_num = len(DAG_list[0])
    start = datetime.datetime.now()

    for t in range(dag_size):
        print(t)
        ancestor_dict = dict()
        for i in range(node_num):
            ancestor_dict[i] = []
    
        adding_err_edge = 0
        deleting_err_edge = 0
        corr_edge = 0

        DAG_test = DAG_list[t]

        true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_test)
                
        count_list.append(find_ancestor.list_diff(true_ancestor_dict, true_ancestor_dict))
        
    
        
        AA_dict = dict()
        for i in range(node_num):
            AA_dict[i] = [i] + true_ancestor_dict[i]
        
        
        
        ancestor_dict_list.append(AA_dict)   
        groups = find_ancestor.get_group(AA_dict, 10000)
        
        M = find_ancestor.get_res(data_list[t], groups, B_list[t])
        
        M = eliminate_loop(M, true_ancestor_dict)
        
        for i in range(len(M)):
            for j in range(len(M)):
                if(M[i][j]):
                    if(DAG_test[i][j]):
                        corr_edge += 1

                    else:
                        adding_err_edge += 1
                else:
                    if(DAG_test[i][j]):
                        deleting_err_edge += 1
    
        adding_list_pruning.append(adding_err_edge)
        deleting_list_pruning.append(deleting_err_edge)
        corr_list_pruning.append(corr_edge)

    end = datetime.datetime.now()

    execute_time = (end - start).seconds
    DAG_count = 0
    DAG_count_3 = 0
    for i in range(len(adding_list_pruning)):
        if (adding_list_pruning[i] + deleting_list_pruning[i]) <= 0:
            DAG_count += 1
        if (adding_list_pruning[i] + deleting_list_pruning[i]) <= 3:
            DAG_count_3 += 1


    TP = np.sum(corr_list_pruning)
    TP_FP = TP + np.sum(adding_list_pruning)
    TP_FN = TP + np.sum(deleting_list_pruning)

    precision = TP / TP_FP
    recall = TP / TP_FN
    F = 2*precision*recall / (precision + recall)

    ancestor_err = 1 - (np.sum(count_list) / (dag_size * node_num))
    
    return precision, recall, F, DAG_count, DAG_count_3, execute_time, ancestor_err


def execute_L(DAG_list, data_list, B_list):
    dag_size = len(DAG_list)

    adding_list_L = list()
    deleting_list_L = list()
    corr_list_L = list()

    node_num = len(DAG_list[0])
    start = datetime.datetime.now()
  
    for t in range(dag_size):
        ancestor_dict = dict()
        for i in range(node_num):
            ancestor_dict[i] = []
        
        DAG_test = DAG_list[t]
        B = B_list[t]
        
        model_L = lingam.DirectLiNGAM()
        model_L.fit(data_list[t])
        m_L = model_L.adjacency_matrix_
       
        adding_err_edge = 0
        deleting_err_edge = 0
        corr_edge = 0

        for i in range(node_num):
            for j in range(node_num):
                if(m_L[i][j]):
                    if(DAG_test[i][j]):
                        corr_edge += 1
                    else:
                        adding_err_edge += 1
                else:
                    if(DAG_test[i][j]):
                        deleting_err_edge += 1

        adding_list_L.append(adding_err_edge)
        deleting_list_L.append(deleting_err_edge)
        corr_list_L.append(corr_edge)

    end = datetime.datetime.now()

    execute_time = (end - start).seconds
    DAG_count = 0
    DAG_count_3 = 0
  
    for i in range(len(adding_list_L)):
        if (adding_list_L[i] + deleting_list_L[i]) <= 0:
            DAG_count += 1
        if (adding_list_L[i] + deleting_list_L[i]) <= 3:
            DAG_count_3 += 1


    TP = np.sum(corr_list_L)
    TP_FP = TP + np.sum(adding_list_L)
    TP_FN = TP + np.sum(deleting_list_L)

    precision = TP / TP_FP
    recall = TP / TP_FN

    F = 2*precision*recall / (precision + recall)

    return precision, recall, F, DAG_count, DAG_count_3, execute_time


def execute_HSIC_test(DAG_list, data_list, B_list, l_alpha, i_alpha, i_alpha_U, p_alpha=1):
    dag_size = len(DAG_list)
    sample_size = len(data_list[0])
    
    ancestor_dict_list = list()
    count_list = list()

    diff_err_list = list()

    node_num = len(DAG_list[0])
    start = datetime.datetime.now()
    
    for t in range(dag_size):
        print(t)
        # temp_list = list()
        ancestor_dict = dict()
        for i in range(node_num):
            ancestor_dict[i] = []
        
        DAG_test = DAG_list[t]

        true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_test)
        
        find_ancestor.get_ancestor_loop_HSIC(data_list[t], ancestor_dict, l_alpha, i_alpha, i_alpha_U, p_alpha)
        
        count_list.append(find_ancestor.list_diff(true_ancestor_dict, ancestor_dict))
        
        AA_dict = dict()
        for i in range(node_num):
            AA_dict[i] = [i] + ancestor_dict[i]
        
        ancestor_dict_list.append(AA_dict)   
        groups = find_ancestor.get_group(AA_dict, sample_size)
        
        M = find_ancestor.get_res(data_list[t], groups, B_list[t])

        M = eliminate_loop(M, ancestor_dict)
        
        err = evaluation.differ_from_ancestor(true_ancestor_dict, ancestor_dict, M)
        if len(diff_err_list) == 0:
            diff_err_list = np.array(err)
        else:
            diff_err_list += np.array(err)
    
    return diff_err_list


def Topo_Sort(G):
    graph = dict() # outdegree
    for i in range(len(G)):
        out_node = list()
        for j in range(len(G)):
            if(G[i][j]):
                out_node.append(j)
            graph[i] = out_node
     
    return topoSort(graph)


def topoSort(graph):     
    in_degrees = dict((u,0) for u in graph)   #初始化所有顶点入度为0  

    num = len(in_degrees)     
    for u in graph:         
        for v in graph[u]:             
            in_degrees[v] += 1    #计算每个顶点的入度     
    Q = [u for u in in_degrees if in_degrees[u] == 0]   # 筛选入度为0的顶点     
    
    Seq = []     
    while Q:  
        # print("Q: ", Q)       
        u = Q.pop()       #默认从最后一个删除         
        Seq.append(u)         
        for v in graph[u]:             
            in_degrees[v] -= 1    #移除其所有出边
            if in_degrees[v] == 0:        
                Q.append(v)          #再次筛选入度为0的顶点
    if len(Seq) == num:       #输出的顶点数是否与图中的顶点数相等
        return True    
    else:         
        return False


def execute_HSIC_DAG(DAG_list, data_list, B_list, l_alpha, i_alpha, i_alpha_U, p_alpha=1):
    dag_size = len(DAG_list)
    sample_size = len(data_list[0])
    
    ancestor_dict_list = list()
    count_list = list()

    node_num = len(DAG_list[0])
    start = datetime.datetime.now()

    is_DAG_count = 0
    is_DAG_after_count = 0
    
    for t in range(dag_size):
        print(t)
        ancestor_dict = dict()
        for i in range(node_num):
            ancestor_dict[i] = []

        DAG_test = DAG_list[t]

        true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_test)
        
        find_ancestor.get_ancestor_loop_HSIC(data_list[t], ancestor_dict, l_alpha, i_alpha, i_alpha_U, p_alpha)
        
        anc_dict = list(ancestor_dict.values())

        count_list.append(find_ancestor.list_diff(true_ancestor_dict, ancestor_dict))
        
    
        
        AA_dict = dict()
        for i in range(node_num):
            AA_dict[i] = [i] + ancestor_dict[i]
                
        ancestor_dict_list.append(AA_dict)   
        groups = find_ancestor.get_group(AA_dict, sample_size)

        M = find_ancestor.get_res(data_list[t], groups, B_list[t])
        
        if Topo_Sort(M):
            is_DAG_count += 1
            is_DAG_after_count += 1
        else:
            M = eliminate_loop(M, anc_dict)
            if Topo_Sort(M):
                is_DAG_after_count += 1
    end = datetime.datetime.now()

    execute_time = (end - start).seconds
    
    return [[is_DAG_count], [is_DAG_after_count]]


def execute_K_DAG(DAG_list, data_list, B_list):
    dag_size = len(DAG_list)

    ancestor_dict_list = list()
    count_list = list()

    node_num = len(DAG_list[0])
    start = datetime.datetime.now()

    is_DAG_count = 0
    is_DAG_after_count = 0
    for t in range(dag_size):
        print(t)
        ancestor_dict = dict()
        for i in range(node_num):
            ancestor_dict[i] = []
        
        # print("epcho: ", t)
        adding_err_edge = 0
        deleting_err_edge = 0
        corr_edge = 0

        DAG_test = DAG_list[t]

        true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_test)
        
        count_list.append(find_ancestor.list_diff(true_ancestor_dict, true_ancestor_dict))
        
    
        
        AA_dict = dict()
        for i in range(node_num):
            AA_dict[i] = [i] + true_ancestor_dict[i]
        
        
        
        ancestor_dict_list.append(AA_dict)   
        groups = find_ancestor.get_group(AA_dict, 10000)

        M = find_ancestor.get_res(data_list[t], groups, B_list[t])
        
        if Topo_Sort(M):
            is_DAG_count += 1

        M = eliminate_loop(M, true_ancestor_dict)
        if Topo_Sort(M):
            is_DAG_after_count += 1


    end = datetime.datetime.now()

    execute_time = (end - start).seconds
    
    return [[is_DAG_count], [is_DAG_after_count]]


def execute_L_DAG(DAG_list, data_list, B_list):
    dag_size = len(DAG_list)

    node_num = len(DAG_list[0])
    start = datetime.datetime.now()
  
    is_DAG_count = 0
    for t in range(dag_size):
        ancestor_dict = dict()
        for i in range(node_num):
            ancestor_dict[i] = []

        DAG_test = DAG_list[t]
        B = B_list[t]

        model_L = lingam.DirectLiNGAM()
        model_L.fit(data_list[t])
        m_L = model_L.adjacency_matrix_
       
        if Topo_Sort(m_L):
            is_DAG_count += 1

    end = datetime.datetime.now()

    execute_time = (end - start).seconds

    return [is_DAG_count]


def Topo_eliminate(G, ancestor_dict):
    graph = dict() # outdegree
    for i in range(len(G)):
        out_node = list()
        for j in range(len(G)):
            if(G[i][j]):
                out_node.append(j)
            graph[i] = out_node

    in_degrees = dict((u,0) for u in graph)   # 初始化所有顶点入度为0  

    num = len(in_degrees)     
    nodes = [i for i in range(num)]
    
    for u in graph:         
        for v in graph[u]:             
            in_degrees[v] += 1    # 计算每个顶点的入度     
    Q = [u for u in in_degrees if in_degrees[u] == 0]   # 筛选入度为0的顶点     
    
    Seq = []     
    while Q:  

        u = Q.pop()       # 默认从最后一个删除         
        Seq.append(u)         
        for v in graph[u]:             
            in_degrees[v] -= 1    # 移除其所有出边
            if in_degrees[v] == 0:        
                Q.append(v)          # 再次筛选入度为0的顶点
    if len(Seq) == num:       # 输出的顶点数是否与图中的顶点数相等
        return G    
    else: 
        unfinished_nodes = [i for i in nodes if i not in Seq]
        for i in unfinished_nodes:
            for j in unfinished_nodes:
                if G[i][j] != 0: # j-->i
                    if j not in ancestor_dict[i]:
                        G[i][j] = 0
    return G


def eliminate_loop(M, ancestor_dict):
    return Topo_eliminate(M, ancestor_dict)
            

def execute_RCD(DAG_list, data_list, B_list, l_alpha, i_alpha):
    dag_size = len(DAG_list)

    adding_list_L = list()
    deleting_list_L = list()
    corr_list_L = list()

    node_num = len(DAG_list[0])
    start = datetime.datetime.now()
    
    for t in range(dag_size):
        ancestor_dict = dict()
        for i in range(node_num):
            ancestor_dict[i] = []
        DAG_test = DAG_list[t]
        B = B_list[t]

        model_RCD = lingam_local.RCD(
                        max_explanatory_num=2
                       , cor_alpha=l_alpha
                       , ind_alpha=i_alpha
                       , shapiro_alpha=1
                    #    , MLHSICR=True
                       )
        
        model_RCD.fit(data_list[t])
        m_L = model_RCD.adjacency_matrix_

        for i in range(node_num):
            for j in range(node_num):
                if np.isnan(m_L[i][j]):
                    m_L[i][j] = 0
       

        adding_err_edge = 0
        deleting_err_edge = 0
        corr_edge = 0


        for i in range(node_num):
            for j in range(node_num):
                if(m_L[i][j]):
                    if(DAG_test[i][j]):
                        corr_edge += 1
                    else:
                        adding_err_edge += 1
                else:
                    if(DAG_test[i][j]):
                        deleting_err_edge += 1

        adding_list_L.append(adding_err_edge)
        deleting_list_L.append(deleting_err_edge)
        corr_list_L.append(corr_edge)

    end = datetime.datetime.now()
    print("start: ", start, " and end: ", end)
    execute_time = (end - start).seconds

    print(execute_time)
    DAG_count = 0
    DAG_count_3 = 0
  
    for i in range(len(adding_list_L)):
        if (adding_list_L[i] + deleting_list_L[i]) <= 0:
            DAG_count += 1
        if (adding_list_L[i] + deleting_list_L[i]) <= 3:
            DAG_count_3 += 1


    TP = np.sum(corr_list_L)
    TP_FP = TP + np.sum(adding_list_L)
    TP_FN = TP + np.sum(deleting_list_L)

    precision = TP / TP_FP
    recall = TP / TP_FN

    F = 2*precision*recall / (precision + recall)

    return precision, recall, F, DAG_count, DAG_count_3, execute_time


def execute_RCD_DAG(DAG_list, data_list, B_list, l_alpha, i_alpha):
    dag_size = len(DAG_list)

    is_DAG_count = 0
    is_DAG_after_count = 0

    node_num = len(DAG_list[0])
    start = datetime.datetime.now()
  
    for t in range(dag_size):
        print(t)
        ancestor_dict = dict()
        for i in range(node_num):
            ancestor_dict[i] = []
        DAG_test = DAG_list[t]
        B = B_list[t]
        model_RCD = lingam_local.RCD(
                        max_explanatory_num=2
                       , cor_alpha=l_alpha
                       , ind_alpha=i_alpha
                       , shapiro_alpha=1
                    #    , MLHSICR=True
                       )
        
        model_RCD.fit(data_list[t])
        m_L = model_RCD.adjacency_matrix_

        ancestor_dict = model_RCD.ancestors_list_

        for i in range(node_num):
            ancestor_dict[i] = list(ancestor_dict[i]) + [i]
       
        for i in range(node_num):
            for j in range(node_num):
                if np.isnan(m_L[i][j]):
                    m_L[i][j] = 0

        if Topo_Sort(m_L):
            is_DAG_count += 1

        m_L = eliminate_loop(m_L, ancestor_dict)
        if Topo_Sort(m_L):
            is_DAG_after_count += 1
    
    return [[is_DAG_count], [is_DAG_after_count]]