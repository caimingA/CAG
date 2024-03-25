import numpy as np
from numba import jit


# @jit(nopython=True)
def get_noisy(T, matrix):
    num = len(matrix)
    noisy = np.random.random(size = (T, len(matrix)))*1*2-1
    return noisy


# @jit(nopython=True)
def lower_triangle_graph(node, edge, max_indegree):
    matrix = np.zeros((node, node))
    visit = np.zeros(node)
    count = 0
    while count != edge:
        edge_set = np.random.randint(low=0, high=node, size=2)
        i = np.max(edge_set)
        j = np.min(edge_set)
        if i == j or matrix[i][j] or visit[i] >= max_indegree:
            continue
        else:
            visit[i] += 1
            matrix[i][j] = 1
            count += 1
  
    # return np.array(matrix)
    return matrix


# @jit(nopython=True)
def get_B_0(matrix):
    num = len(matrix)
    B = np.random.random(size = (num, num))
    B = np.where(B > 0.5, B, B - 1)
    B = np.where(matrix, B, 0)
    return B


# @jit(nopython=True)
def get_x(matrix, noisy, B_0):
    I = np.identity(len(B_0))
    x_1 = (np.linalg.pinv(I - B_0)).dot(noisy.T)
    return x_1


# @jit(nopython=True)
def generate_DAGs(node_num, edge_num, max_indegree, sample_size, dag_size):  

    DAG_list = list()
    B_list = list()
    noisy_list = list()
    data_list = list()
    
    for t in range(dag_size):
        DAG_temp = lower_triangle_graph(node_num, edge_num, max_indegree)
        DAG_list.append(DAG_temp)

        B_temp =  get_B_0(DAG_temp)
        B_list.append(B_temp)

        noisy_temp = get_noisy(sample_size, DAG_temp)
        noisy_list.append(noisy_temp)

        x_list_temp = get_x(DAG_temp, noisy_temp, B_temp).T
        data_list.append(x_list_temp)

    return DAG_list, data_list, B_list
