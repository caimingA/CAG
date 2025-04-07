import numpy as np
import data_generator
import graphviz

import itertools
import draw
import myLiNGAM
import find_ancestor
import time
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
import evaluation
import execute

import lingam
import lingam_local

import SADA
import CAPA


# def draw_graphviz(graph):


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
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
)

# DAG_temp = np.array([
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
#     ]
# )

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
node_num = 3
# DAG_temp = data_generator.lower_triangle_graph(10, 9, 1)
# # for i in range(100):
# #     print(np.random.uniform(0, 1))

sample_size =1000
# node_num = 10
B_temp = data_generator.get_B_0(DAG_temp)
draw.draw(B_temp)
print(DAG_temp)
noisy_temp = data_generator.get_noisy(sample_size, DAG_temp)
# print(noisy_temp)

x_list_temp = data_generator.get_x(DAG_temp, noisy_temp, B_temp).T
# x_list_temp_re = x_list_temp[: , 1: ]
print(x_list_temp.shape)
# print(x_list_temp_re.shape)

# g, edges = fci(x_list_temp_re)

cg = pc(x_list_temp)

print(cg.G.graph)
# print(g)
# print(g.graph)

# pdy = GraphUtils.to_pydot(g)
# pdy.write_png('simple_test.png')
# cg.draw_pydot_graph()
# draw.draw(B_temp)
