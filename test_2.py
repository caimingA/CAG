import numpy as np
import data_generator

import itertools
import draw
import myLiNGAM
import find_ancestor
import time
# from causallearn.search.ConstraintBased.PC import pc
# from causallearn.search.ConstraintBased.FCI import fci
import evaluation
import execute

import lingam
import lingam_local

import SADA
import CAPA

# import hyppo

# print(set())

# DAG_temp = np.array([
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
#     ]
# )

# DAG_temp = np.array([
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
#     ]
# )

DAG_temp = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)

# DAG_temp = np.array([
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#         # [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         # [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         # [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         # [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     ]
# )


# DAG_temp = np.array([
#         [0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 1.0, 0.0],
#     ]
# )

# DAG_temp = np.array([
#         [0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0],
#     ]
# )
node_num = 10
# DAG_temp = data_generator.lower_triangle_graph(10, 9, 1)
# # for i in range(100):
# #     print(np.random.uniform(0, 1))
# for i in range(0, node_num):
#     for j in range(i + 1, node_num):
#         print(i, " and ", j)


sample_size =1000
# node_num = 10
B_temp =  data_generator.get_B_0(DAG_temp)
draw.draw(B_temp)
print(DAG_temp)
noisy_temp = data_generator.get_noisy(sample_size, DAG_temp)
# print(noisy_temp)

x_list_temp = data_generator.get_x(DAG_temp, noisy_temp, B_temp).T
print(x_list_temp.shape)

print("============CAPA============")
time_1 = time.time()
print(CAPA.CAPA(x_list_temp, 2, 0.001))

time_2 = time.time()

# print("============Proposed============")
# ancestor_dict = dict()
# for i in range(node_num):
#     ancestor_dict[i] = []

# # # find_ancestor.get_ancestor_loop_HSIC(x_list_temp, ancestor_dict, 0.01, 0.001, 0.001)
# find_ancestor.get_ancestor_loop_KCI_2(x_list_temp, ancestor_dict, 0.01, 0.001, 0.001)

# print(ancestor_dict)
# AA_dict = dict()
# for i in range(node_num):
#     AA_dict[i] = [i] + ancestor_dict[i]
# # ancestor_dict_list.append(AA_dict)   
# groups = find_ancestor.get_group(AA_dict, sample_size)

# print(groups)
# time_3 = time.time()

print("============Proposed============")
ancestor_dict = dict()
for i in range(node_num):
    ancestor_dict[i] = []

# # find_ancestor.get_ancestor_loop_HSIC(x_list_temp, ancestor_dict, 0.01, 0.001, 0.001)
find_ancestor.get_ancestor_loop_KCI(x_list_temp, ancestor_dict, 0.01, 0.001, 0.001)

print(ancestor_dict)
AA_dict = dict()
for i in range(node_num):
    AA_dict[i] = [i] + ancestor_dict[i]
# ancestor_dict_list.append(AA_dict)   
groups = find_ancestor.get_group(AA_dict, sample_size)

print(groups)
time_3 = time.time()

ancestor_dict = dict()
for i in range(node_num):
    ancestor_dict[i] = []

# # find_ancestor.get_ancestor_loop_HSIC(x_list_temp, ancestor_dict, 0.01, 0.001, 0.001)
find_ancestor.get_ancestor_loop_HSIC(x_list_temp, ancestor_dict, 0.01, 0.001, 0.001)

print(ancestor_dict)
AA_dict = dict()
for i in range(node_num):
    AA_dict[i] = [i] + ancestor_dict[i]
# ancestor_dict_list.append(AA_dict)   
groups = find_ancestor.get_group(AA_dict, sample_size)

print(groups)

# print("============SADA============")
# print(SADA.SADA(x_list_temp, 0.001))

time_4 = time.time()

print(time_2 - time_1)
print(time_3 - time_2)
print(time_4 - time_3)
