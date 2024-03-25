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
#         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
#     ]
# )

DAG_temp = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ]
)

# DAG_temp = np.array([
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     ]
# )

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
#         [0.0, 1.0, 0.0],
#         [0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0],
        
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

sample_size =1000
# node_num = 10
B_temp =  data_generator.get_B_0(DAG_temp)
draw.draw(B_temp)
print(DAG_temp)
noisy_temp = data_generator.get_noisy(sample_size, DAG_temp)
# print(noisy_temp)

x_list_temp = data_generator.get_x(DAG_temp, noisy_temp, B_temp).T
print(x_list_temp.shape)

time_1 = time.time()
print(CAPA.CAPA(x_list_temp, 1, 0.001))
print("=======")
time_2 = time.time()
# print(SADA.SADA(x_list_temp))
# for i in itertools.combinations({1, 2, 3, 4, 5}, 1):
#     print(set(i))f

# model = lingam.DirectLiNGAM()
# model_RCD = lingam_local.RCD(
#         max_explanatory_num=2
#        , cor_alpha=0.01
#        , ind_alpha=0.001
#        , shapiro_alpha=1
#     #    , MLHSICR=True
#     )

# model_RCD.fit(x_list_temp)
# m_L = model_RCD.adjacency_matrix_

# for i in range(node_num):
#     for j in range(node_num):
#         if np.isnan(m_L[i][j]):
#             m_L[i][j] = 0

# adding_err_edge_2 = 0
# deleting_err_edge_2 = 0
# corr_edge_2 = 0
# # draw.draw()
# for i in range(len(m_L)):
#     for j in range(len(m_L)):
#         if(m_L[i][j]):
#             if(DAG_temp[i][j]):
#                 corr_edge_2 += 1
#             else:
#                 adding_err_edge_2 += 1
#         else:
#             if(DAG_temp[i][j]):
#                 deleting_err_edge_2 += 1

# ancestor_dict_RCD = model_RCD._ancestors_list
# true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_temp)

# Anc_N = 0
# for i in true_ancestor_dict:
#     Anc_N += len(i)

# count = find_ancestor.list_diff(true_ancestor_dict, ancestor_dict_RCD)


# pre = np.true_divide(corr_edge_2, corr_edge_2 + adding_err_edge_2)
# rec = np.true_divide(corr_edge_2, corr_edge_2 + deleting_err_edge_2)
# # corr_edge_2/(corr_edge_2 + deleting_err_edge_2)
# print(corr_edge_2, ", ", adding_err_edge_2
#       , ", ", deleting_err_edge_2
#       , ", pre = ", pre
#       , ", rec = ", rec
#       , ", F = ", np.true_divide(2*pre*rec, (pre + rec)))

# temp_loop = find_ancestor.find_loop(m_L)

# # print(count)
# print(model_RCD._ancestors_list)
# print(true_ancestor_dict)
# print(count)
# print("---------------------")
# # count = 0
# while(len(temp_loop)):
#     print(True)
#     m_L = find_ancestor.wald_eliminate_loop(m_L, temp_loop, x_list_temp)
#     temp_loop = find_ancestor.find_loop(m_L)


# adding_err_edge_2 = 0
# deleting_err_edge_2 = 0
# corr_edge_2 = 0
# # draw.draw()
# for i in range(len(m_L)):
#     for j in range(len(m_L)):
#         if(m_L[i][j]):
#             if(DAG_temp[i][j]):
#                 corr_edge_2 += 1
#             else:
#                 adding_err_edge_2 += 1
#         else:
#             if(DAG_temp[i][j]):
#                 deleting_err_edge_2 += 1

# pre = np.true_divide(corr_edge_2, corr_edge_2 + adding_err_edge_2)
# rec = np.true_divide(corr_edge_2, corr_edge_2 + deleting_err_edge_2)
# # corr_edge_2/(corr_edge_2 + deleting_err_edge_2)
# print(corr_edge_2, ", ", adding_err_edge_2
#       , ", ", deleting_err_edge_2
#       , ", pre = ", pre
#       , ", rec = ", rec
#       , ", F = ", np.true_divide(2*pre*rec, (pre + rec)))
# draw.draw(m_L)

ancestor_dict = dict()
for i in range(node_num):
    ancestor_dict[i] = []

# # find_ancestor.get_ancestor_loop_HSIC(x_list_temp, ancestor_dict, 0.01, 0.001, 0.001)
find_ancestor.get_ancestor_loop_KCI(x_list_temp, ancestor_dict, 0.01, 0.001, 0.001)


AA_dict = dict()
for i in range(node_num):
    AA_dict[i] = [i] + ancestor_dict[i]
# ancestor_dict_list.append(AA_dict)   
groups = find_ancestor.get_group(AA_dict, sample_size)

# print(groups)
time_3 = time.time()

# print(SADA.SADA(x_list_temp, 0.001))

time_4 = time.time()

print(time_2 - time_1)
print(time_3 - time_2)
print(time_4 - time_3)
# M_res = find_ancestor.get_res(x_list_temp, groups, B_temp, ancestor_dict)

# print(M_res[0])



# # temp_loop = find_ancestor.find_loop(M_res)
# # # print(model_RCD._ancestors_list)
# # # count = 0
# # while(len(temp_loop)):
# #     print(True)
# #     M_res = find_ancestor.wald_eliminate_loop(M_res, temp_loop, x_list_temp)
# #     temp_loop = find_ancestor.find_loop(M_res)


# adding_err_edge_2 = 0
# deleting_err_edge_2 = 0
# corr_edge_2 = 0
# # draw.draw()
# for i in range(len(M_res[0])):
#     for j in range(len(M_res[0])):
#         if(M_res[0][i][j]):
#             if(DAG_temp[i][j]):
#                 corr_edge_2 += 1
#             else:
#                 adding_err_edge_2 += 1
#         else:
#             if(DAG_temp[i][j]):
#                 deleting_err_edge_2 += 1

# pre = corr_edge_2/(corr_edge_2 + adding_err_edge_2)
# rec = corr_edge_2/(corr_edge_2 + deleting_err_edge_2)
# print(corr_edge_2, ", ", adding_err_edge_2
#       , ", ", deleting_err_edge_2
#       , ", pre = ", corr_edge_2/(corr_edge_2 + adding_err_edge_2)
#       , ", rec = ", corr_edge_2/(corr_edge_2 + deleting_err_edge_2)
#       , ", F = ", 2*pre*rec / (pre + rec))

# res_temp = evaluation.differ_from_ancestor(true_ancestor_dict, ancestor_dict, M_res[0])
# print(Anc_N)

# print(res_temp)

# # print(res_temp[0] / res_temp[5])
# # print(res_temp[1] / res_temp[4])

# estimated_DAG = find_ancestor.get_true_ancestor(M_res[0])
# print(true_ancestor_dict, "\n", estimated_DAG, "\n", ancestor_dict, "\n", ancestor_dict_RCD)

# draw.draw(M_res[0])

# print(M_res)
# draw.draw(M_res[0])
# draw.draw(M_res[1])
# draw.draw(M_res[2])
# sart = time.time()
# data_generator_no.generate_DAGs(10, 9, 1, 100, 100)

# point_1 = time.time()

# data_generator.generate_DAGs(10, 9, 1, 100, 100)

# point_2 = time.time()

# print(point_1 - start)
# print(point_2 - point_1)
# for i in range(2):
#     if i == 0:
#         x_list_temp = np.dot(noisy_temp, B_temp)
#     else:
#         x_list_temp = np.dot(x_list_temp, B_temp)
    # x_0 = noisy_temp[:, 0]
    # x_1 = noisy_temp[:, 1]
    # x_2 = noisy_temp[:, 2]
    # if i == 0:
    #     # x_0 = noisy_temp[:, 0]
    #     x_1 += B_temp[1][0] * x_0
    #     x_2 += B_temp[2][1] * x_1
    # else:
    #     x_0 = B_temp[0][2]*x_2 + noisy_temp[:, 0]
    #     x_1 = B_temp[1][0] * x_0 + noisy_temp[:, 1]
    #     x_2 = B_temp[2][1] * x_1 + noisy_temp[:, 2]

# x_list_temp = np.array([x_0, x_1, x_2]).T

# x_list_temp = data_generator.get_x(DAG_temp, noisy_temp, B_temp).T
# print(x_list_temp.shape)
# true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_temp)
# xx_list_temp = x_list_temp[:, 1:]
# print(B_temp)

# print(x_list_temp.shape)
# print(true_ancestor_dict)

# draw.draw(DAG_temp)
# draw.draw(DAG_test)

# DAG_res = execute.eliminate_loop(DAG_test, true_ancestor_dict)

# draw.draw(DAG_res)
# print(xx_list_temp.shape)

# t = time.strftime("%Y-%m-%d", time.localtime())
# print(type(t))
# print(DAG_temp)

# draw.draw(DAG_temp)

# print(np.round(myLiNGAM.myDirectLiNGAM(x_list_temp)))
# draw.draw(myLiNGAM.myDirectLiNGAM(x_list_temp))
# cg = pc(x_list_temp)
# Markov_temp = cg.G.graph
# print(Markov_temp)


# G, edges = fci(xx_list_temp)
# print(G)

# Markov_temp = np.array([
#         [0.0, 1.0, -1.0, 0.0],
#         [1.0, 0.0, 0.0, -1.0],
#         [-1.0, 0.0, 0.0, -1.0],
#         [0.0, 1.0, 1.0, 0.0],
#     ]
# )

# draw.draw_Markov(Markov_temp)
# cg.draw_pydot_graph()

# true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_temp)

# ancestor_dict = dict()
# for i in range(3):
#     ancestor_dict[i] = []

# find_ancestor.get_ancestor_loop_HSIC(x_list_temp, ancestor_dict, 0.01, 0.001, 0.1)

# print(ancestor_dict)
# M_temp = np.zeros_like(Markov_temp)
# for i in range(1, len(Markov_temp)):
#     for j in range(0, i):
#         if Markov_temp[i][j] == -1 and Markov_temp[j][i] == -1:
#             if j in ancestor_dict[i]:
#                 M_temp[i][j] = 1
#         if Markov_temp[i][j] == -1 and Markov_temp[j][i] == 1:
#             M_temp[j][i] = 1
#         if Markov_temp[j][i] == -1 and Markov_temp[i][j] == 1:
#             M_temp[i][j] = 1
# print(M_temp)

# draw.draw(M_temp)

# node_num = len(DAG_temp)
# AA_dict = dict()
# for i in range(node_num):
#     AA_dict[i] = [i] + ancestor_dict[i]

# groups = find_ancestor.get_group(AA_dict, sample_size)

# print(ancestor_dict)
# res = find_ancestor.get_res_PC(x_list_temp, groups, DAG_temp, ancestor_dict)
# print(res)
# print(Markov_temp)
# print(groups)
# M = find_ancestor.get_res(x_list_temp, groups, B_temp)
# draw.draw(M)
# M = np.array([
#         [0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0],
#     ]
# )
# ancestor_dict = {
#     0: [],
#     1: [],
#     2: [],
#     3: []
# }


# print(M)

# draw.draw(M)
# M_no = execute.eliminate_loop(M, ancestor_dict)
# print(M_no)
# draw.draw(M_no)

# model_rcd = lingam.RCD(
#                         max_explanatory_num=2
#                        , cor_alpha=0.1
#                        , ind_alpha=0.3
#                        , shapiro_alpha=1
#                     #    , MLHSICR=True
#                        )

# DAG_temp = np.array([
#         [0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0, 0.0],
#     ]
# )

# draw.draw(DAG_temp)

# M = execute.Topo_eliminate(DAG_temp, [[], [0], [0, 1], [0, 1, 2]])

# draw.draw(M)
# draw.draw(DAG_temp)
# noisy_temp = data_generator.get_noisy(sample_size, DAG_temp)
# x_list_temp = data_generator.get_x(DAG_temp, noisy_temp, DAG_temp).T

# true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_temp)

# model_rcd.fit(x_list_temp)

# M_RCD = model_rcd.adjacency_matrix_

# print(model_rcd.ancestors_list_)
# print(M_RCD)
# draw.draw(M_RCD)

# for i in range(len(M_RCD)):
#     for j in range(len(M_RCD)):
#         # print(type(M_RCD[i][j]))
#         if np.isnan(M_RCD[i][j]):
#         # if M_RCD[i][j] == np.nan:
#             M_RCD[i][j] = 0

# print(M_RCD)            
# draw.draw(M_RCD)

# M = execute.Topo_eliminate(M_RCD, model_rcd.ancestors_list_)
# draw.draw(M)
# err = evaluation.differ_from_ancestor(true_ancestor_dict, ancestor_dict, M)
# print(err)
# print(ancestor_dict)
# print(true_ancestor_dict)

# draw.draw(res)

