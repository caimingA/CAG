import numpy as np
import data_generator
import draw
import myLiNGAM
import find_ancestor
import time
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
import evaluation
import execute

import lingam
import lingam_local

DAG_temp = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        
    ]
)

sample_size = 1000
B_temp =  data_generator.get_B_0(DAG_temp)
noisy_temp = data_generator.get_noisy(sample_size, DAG_temp)
# print(noisy_temp.shape)

x_list_temp = data_generator.get_x(DAG_temp, noisy_temp, B_temp).T
# print(x_list_temp.shape)
true_ancestor_dict = find_ancestor.get_true_ancestor(DAG_temp)

print("true ancestor: ", true_ancestor_dict)

draw.draw(DAG_temp)

ancestor_dict = dict()
for i in range(3):
    ancestor_dict[i] = []

find_ancestor.get_ancestor_loop_HSIC_to_RCD(x_list_temp, ancestor_dict, 0.01, 0.001, 0.1)

print("esitmated ancestors by proposed method: ", ancestor_dict)

node_num = len(DAG_temp)
AA_dict = dict()
for i in range(node_num):
    AA_dict[i] = [i] + ancestor_dict[i]

print("grouping result: ", end="")
groups = find_ancestor.get_group(AA_dict, sample_size)

M = find_ancestor.get_res(x_list_temp, groups, B_temp)
draw.draw(M)


model_rcd = lingam_local.RCD(
                        max_explanatory_num=2
                       , cor_alpha=0.01
                       , ind_alpha=0.001
                       , shapiro_alpha=1
                    #    , MLHSICR=True
                       )

model_rcd.fit(x_list_temp)

M_RCD = model_rcd.adjacency_matrix_

print("esitmated ancestors by rcd: ", model_rcd.ancestors_list_)
print("B matrix estimated by rcd")
print(M_RCD)
draw.draw(M_RCD)
